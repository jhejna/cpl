import datetime
import functools
import os
import random
import shutil
import tempfile
from typing import Callable, Dict, List, Optional, Union

import gym
import torch

from research.utils import utils

from . import sampling, storage


def remove_stack_dim(space: gym.Space) -> gym.Space:
    if isinstance(space, gym.spaces.Dict):
        return gym.spaces.Dict({k: remove_stack_dim(v) for k, v in space.items()})
    elif isinstance(space, gym.spaces.Box):
        return gym.spaces.Box(low=space.low[1:], high=space.high[1:])
    else:
        return space


class ReplayBuffer(torch.utils.data.IterableDataset):
    """
    Generic Replay Buffer Class.

    This class adheres to the following conventions to support multiprocessing:
    1. Variables/functions starting with "_", like "_help" are to be used only by the replay buffer internaly. They
        are carefully setup for multiprocesing.
    2. variables/functions named regularly without a leading "_" are to be used by the main thread. This includes
        standard functions like "add".

    There are a few critical setup options.
    1. Capacity: determines if the buffer is setup upon creation. If it is set to a known value, then we can add data
        online with `add`, or by pulling more data from disk. If is set to None, the dataset is initialized to the full
        size of the offline dataset.
    2. path: path to offline data that will be loaded
    3. _data_generator

    Some options are mutually exclusive. For example, it is bad to use a non-distributed layout with
    workers and online data. This will generate a bunch of copy on writes.

    Data is expected to be stored in a "next" format. This means that data is stored like this:
    s_0, dummy, dummy, dummy
    s_1, a_0  , r_0  , d_0
    s_2, a_1  , r_1  , d_1
    s_3, a_2  , r_2  , d_2 ... End of episode!
    s_0, dummy, dummy, dummy
    s_1, a_0  , r_0  , d_0
    s_2, a_1  , r_1  , d_1

    This format is expected from the load(path) funciton.

    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        sample_fn: Union[str, Callable] = "sample",
        sample_kwargs: Optional[Dict] = None,
        epoch_ratio: float = 1.0,
        path: Optional[str] = None,
        capacity: Optional[int] = None,
        exclude_keys: Optional[List[str]] = None,
        include_keys: Optional[Dict] = None,
        stacked_obs: bool = False,
        stacked_action: bool = False,
        distributed: bool = False,
        fetch_every: int = 1000,
        cleanup: bool = True,
    ) -> None:
        # Remove stacking if present.
        self.stacked_obs = stacked_obs
        if self.stacked_obs:
            observation_space = remove_stack_dim(observation_space)
        self.stacked_action = stacked_action
        if self.stacked_action:
            action_space = remove_stack_dim(action_space)

        self.observation_space = observation_space
        self.action_space = action_space

        # Construct the space for the buffer
        self.exclude_keys = [] if exclude_keys is None else exclude_keys  # keys to exclude in the storage buffer
        buffer_space = {
            "obs": self.observation_space,
            "action": self.action_space,
            "reward": 0.0,
            "done": False,
            "discount": 1.0,
        }
        flattened_buffer_space = utils.flatten_dict(buffer_space)
        if include_keys is not None:
            flattened_buffer_space.update(include_keys)
            print("FLATTENED BUFFER SPACE", flattened_buffer_space)
        for k in self.exclude_keys:
            if k in flattened_buffer_space:
                del flattened_buffer_space[k]
        self.buffer_space = utils.nest_dict(flattened_buffer_space)

        self.dummy_action = self.action_space.sample()
        self.capacity = capacity

        # Setup the sampler
        if isinstance(sample_fn, str):
            sample_fn = vars(sampling)[sample_fn]
        # Use functools partial to override the default args.
        sample_kwargs = {} if sample_kwargs is None else sample_kwargs
        self.sample_fn = functools.partial(sample_fn, **sample_kwargs)
        # Add sampling parameters
        self.epoch_ratio = epoch_ratio

        # Path for preloaded data
        self.path = path

        # Setup based on distributed value
        self.distributed = distributed
        if self.distributed:
            self.cleanup = cleanup
            self.fetch_every = fetch_every
            if self.capacity is not None:
                self.storage_path = tempfile.mkdtemp(prefix="replay_buffer_")
                print("[research] Replay Buffer Storage Path", self.storage_path)
                self.current_ep = utils.nest_dict({k: list() for k in flattened_buffer_space.keys()})
            self.num_episodes = 0
        else:
            self._alloc(self.capacity)  # Alloc immediately

    def _alloc(self, capacity):
        # Create the data generator
        self._current_data_generator = self._data_generator()

        if capacity is None:
            # Allocte the entire dataset
            data = utils.concatenate(*list(self._current_data_generator), dim=0)
            self._storage = storage.FixedStorage(data)
        else:
            # Construct the buffer space. Remember to exclude any exclude keys
            self._storage = storage.CircularStorage(self.buffer_space, capacity)
            # Fill the storage.
            # if self.path is not None:
            for data in self._current_data_generator:
                self._storage.extend(data)
                if self._storage.size >= self._storage.capacity:
                    break

        print("[ReplayBuffer] Allocated {:.2f} GB".format(self._storage.bytes / 1024**3))

    def _data_generator(self):
        """
        Can be overridden in order to load the initial data differently.
        By default assumes the data to be the standard format, and returned as a data dictionary.
        or
        None

        This function can be overriden by sub-classes in order to produce data batches.
        It should do the following:
        1. split data across torch data workers
        2. randomize the order of data
        3. yield data of the form dicts
        """
        if self.path is None:
            return

        # By default get all of the file names that are distributed at the correct index
        worker_info = torch.utils.data.get_worker_info()
        num_workers = 1 if worker_info is None else worker_info.num_workers
        worker_id = 0 if worker_info is None else worker_info.id

        ep_filenames = [os.path.join(self.path, f) for f in os.listdir(self.path) if f.endswith(".npz")]
        random.shuffle(ep_filenames)  # Shuffle all the filenames

        if num_workers > 1 and len(ep_filenames) == 1:
            print(
                "[ReplayBuffer] Warning: using multiple workers but single replay file. Reduce memory usage by sharding"
                " data with `save` instead of `save_flat`."
            )
        elif num_workers > 1 and len(ep_filenames) < num_workers:
            print("[ReplayBuffer] Warning: using more workers than dataset files.")

        for ep_filename in ep_filenames:
            ep_idx, _ = [int(x) for x in os.path.splitext(ep_filename)[0].split("_")[-2:]]
            # Spread loaded data across workers if we have multiple workers and files.
            if ep_idx % num_workers != worker_id and len(ep_filenames) > 1:
                continue  # Only yield the files belonging to this worker.
            data = storage.load_data(ep_filename, exclude_keys=self.exclude_keys)
            yield data

    def _fetch_offline(self) -> int:
        """
        This simple function fetches a new episode from the offline dataset and adds it to the buffer.
        This is done for each worker.
        """
        try:
            data = next(self._current_data_generator)
        except StopIteration:
            self._current_data_generator = self._data_generator()
            data = next(self._current_data_generator)
        self._storage.extend(data)
        # Return the fetched size
        return len(data["done"])  # data must have the done key for storage

    def _fetch_online(self) -> int:
        worker_info = torch.utils.data.get_worker_info()
        assert worker_info is not None, "Must use distributed buffer for online fetching."

        ep_filenames = sorted([os.path.join(self.storage_path, f) for f in os.listdir(self.storage_path)], reverse=True)
        fetched_size = 0
        for ep_filename in ep_filenames:
            ep_idx, ep_len = [int(x) for x in os.path.splitext(ep_filename)[0].split("_")[-2:]]
            if ep_idx % worker_info.num_workers != worker_info.id:
                continue
            if ep_filename in self._episode_filenames:
                break  # We found something we have already loaded
            if fetched_size + ep_len > self._storage.capacity:
                break  # do not fetch more than the size of the replay buffer

            data = storage.load_data(ep_filename, exclude_keys=self.exclude_keys)
            self._storage.extend(data)
            self._episode_filenames.add(ep_filename)
            if self.cleanup:
                try:
                    os.remove(ep_filename)
                except OSError:
                    pass

        return fetched_size

    def _get_dummy_transition(self, obs):
        flattened_buffer_space = utils.flatten_dict(self.buffer_space)
        dummy_transition = {
            k: v.sample() if isinstance(v, gym.Space) else v
            for k, v in flattened_buffer_space.items()
            if not k.startswith("obs") and not k.startswith("action")
        }
        dummy_transition = utils.nest_dict(dummy_transition)
        dummy_transition["obs"] = obs
        dummy_transition["action"] = self.dummy_action
        return dummy_transition

    def _reset_current_ep(self):
        ep_idx = self.num_episodes
        ep_len = len(self.current_ep["done"])
        self.num_episodes += 1
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        ep_filename = f"{ts}_{ep_idx}_{ep_len}.npz"
        storage.save_data(self.current_ep, os.path.join(self.storage_path, ep_filename))

        flattened_buffer_space = utils.flatten_dict(self.buffer_space)
        ep = {k: list() for k in flattened_buffer_space.keys()}
        self.current_ep = utils.nest_dict(ep)

    def add(self, **kwargs):
        assert self.capacity is not None, "Tried to extend to a static size buffer."
        # Preprocess here before adding to storage
        if len(kwargs) == 1:
            assert "obs" in kwargs
            kwargs = self._get_dummy_transition(kwargs["obs"])
            if self.stacked_obs:
                kwargs["obs"] = utils.get_from_batch(kwargs["obs"], -1)
        else:
            # We have a full transitions
            if self.stacked_obs:
                kwargs["obs"] = utils.get_from_batch(kwargs["obs"], -1)
            if self.stacked_action:
                kwargs["action"] = utils.get_from_batch(kwargs["action"], -1)

        assert "done" in kwargs, "Need done key for ReplayBuffer"

        # This function is overwritten for distributed / local buffers
        if self.distributed:
            # Add to the current thread, and dump to disk
            utils.append(self.current_ep, kwargs)
            if kwargs["done"]:
                self._reset_current_ep()
        else:
            # Add directly
            self._learning_online = True
            self._storage.add(kwargs)

    def extend(self, **kwargs):
        assert "done" in kwargs, "Need done key for ReplayBuffer"
        assert self.capacity is not None, "Tried to extend to a static size buffer."
        # TODO: There is a chance that if we add a full sequence we will end up with (B, T, stack, ...)
        # which is not what we want. We could compare the shapes of the observation space to fix it
        # but this code might be unnecesary, as this class shouldn't really be used like that anyways.
        if self.distributed:
            # Add to the current thread, and dump to disk
            utils.extend(self.current_ep, kwargs)
            if kwargs["done"][-1]:
                self._reset_current_ep()
        else:
            # Add directly
            self._learning_online = True
            self._storage.extend(kwargs)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        if self.distributed:
            if self.cleanup:
                print("[research] Warning, attempting to save a cleaned up replay buffer. There are likely no files")
            srcs = os.listdir(self.storage_path)
            for src in srcs:
                shutil.move(os.path.join(self.storage_path, src), os.path.join(path, src))
            print("Successfully saved", len(srcs), "episodes.")
        else:
            ep_len = self._storage.size
            ep_idx = 0
            ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
            ep_filename = f"{ts}_{ep_idx}_{ep_len}.npz"
            save_path = os.path.join(path, ep_filename)
            self._storage.save(save_path)

    def sample(self, *args, **kwargs):
        return self.sample_fn(self._storage, *args, **kwargs)

    def __iter__(self):
        assert not hasattr(self, "_iterated"), "__iter__ called twice!"
        self._iterated = True
        worker_info = torch.utils.data.get_worker_info()
        assert (worker_info is not None) == self.distributed, "ReplayBuffer.distributed not set correctly!"

        # allocate the buffer with the given capacity
        if self.distributed:
            self._alloc(None if self.capacity is None else self.capacity // worker_info.num_workers)
            self._episode_filenames = set()

        self._learning_online = False

        samples_since_last_offline_fetch = 0
        samples_since_last_online_fetch = 0
        last_offline_fetch_size = 0

        batch_size = self.sample_fn.keywords.get("batch_size", 1)
        stack_size = self.sample_fn.keywords.get("stack", 1)
        seq_size = self.sample_fn.keywords.get("seq_length", 1)

        while True:
            if self._storage.size < seq_size * stack_size + 1:
                yield {}  # If the buffer is too small for sampling, continue.
            else:
                sample = self.sample_fn(self._storage)
                if batch_size == 1:
                    sample = utils.squeeze(sample, 0)
                yield sample

            # Fetch new data if we have a circular buffer.
            if isinstance(self._storage, storage.CircularStorage):
                if self.distributed:  # Always check for online data
                    # We fetch from the online buffer
                    samples_since_last_online_fetch += 1
                    if samples_since_last_online_fetch >= self.fetch_every:
                        fetch_size = self._fetch_online()
                        self._learning_online = self._learning_online or (fetch_size > 0)
                        samples_since_last_online_fetch = 0

                if not self._learning_online and self.path is not None:
                    # We fetch from the offline buffer
                    samples_since_last_offline_fetch += 1
                    data_pts_since_last_offline_fetch = (
                        samples_since_last_offline_fetch * batch_size * seq_size * stack_size
                    )
                    if data_pts_since_last_offline_fetch >= last_offline_fetch_size * self.epoch_ratio:
                        last_offline_fetch_size = self._fetch_offline()
                        samples_since_last_offline_fetch = 0

    def __del__(self):
        if not self.distributed:
            return
        if self.cleanup:
            return
        else:
            paths = [os.path.join(self.storage_path, f) for f in os.listdir(self.storage_path)]
            for path in paths:
                try:
                    os.remove(path)
                except OSError:
                    pass
            try:
                os.rmdir(self.storage_path)
            except OSError:
                pass
