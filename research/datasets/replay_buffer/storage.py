import abc
import io
import os
from collections.abc import Iterable
from typing import Dict, List, Optional, Union

import gym
import numpy as np

from research.utils import utils


def load_data(path: str, exclude_keys: Optional[List[str]]) -> Dict:
    with open(path, "rb") as f:
        data = np.load(f)
        data = {k: data[k] for k in data.keys()}
    # Unnest the data to get everything in the correct format
    for k in exclude_keys:
        if k in data:
            del data[k]  # Remove exclude keys
    data = utils.nest_dict(data)
    return data


def save_data(data: Dict, path: str) -> None:
    # Flatten everything for saving as an np array
    data = utils.flatten_dict(data)
    # Format everything into numpy in case it was saved as a list
    for k in data.keys():
        if isinstance(data[k], np.ndarray) and not data[k].dtype == np.float64:  # Allow float64 carve out.
            continue
        elif isinstance(data[k], list):
            first_value = data[k][0]
            if isinstance(first_value, (np.float64, float)):
                dtype = np.float32  # Detect and convert out float64
            elif isinstance(first_value, (np.ndarray, np.generic)):
                dtype = first_value.dtype
            elif isinstance(first_value, bool):
                dtype = np.bool_
            elif isinstance(first_value, int):
                dtype = np.int64
            data[k] = np.array(data[k], dtype=dtype)
        else:
            raise ValueError("Unsupported type passed to `save_data`.")

    assert len(set(map(len, data.values()))) == 1, "All data keys must be the same length."

    with io.BytesIO() as bs:
        np.savez_compressed(bs, **data)
        bs.seek(0)
        with open(path, "wb") as f:
            f.write(bs.read())


def get_bytes(buffer: Union[Dict, np.ndarray]) -> int:
    if isinstance(buffer, dict):
        return sum([get_bytes(v) for v in buffer.values()])
    elif isinstance(buffer, np.ndarray):
        return buffer.nbytes
    else:
        raise ValueError("Unsupported type passed to `get_bytes`.")


class Storage(abc.ABC):
    """
    The storage object is responsible for holding the data.
    In a distributed setup, each worker might have its own storage object that holds data.

    All storage objects must be given a "done" flag.
    This is used to derive the starts, ends, and lengths properties.
    Done should be true at the last step of the episode.
    """

    @property
    def capacity(self):
        return self._capacity

    @property
    def size(self):
        return self._size

    @property
    def starts(self):
        return self._starts

    @property
    def ends(self):
        return self._ends

    @property
    def lengths(self):
        return self._lengths

    @property
    def bytes(self):
        return get_bytes(self._buffers)

    def save(self, path):
        """
        Directly save the buffer Storage. This saves everything as a flat file.
        This is generally not a good idea as it creates gigantic files.
        """
        assert self._size != 0, "Trying to save Storage with no data."
        assert path.endswith(".npz"), "Path given to `save` was bad. Must save in .npz format."
        data = utils.get_from_batch(self._buffers, 0, self._size)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return save_data(data, path)  # Returns the save path

    def __getitem__(self, key):
        return self._buffers[key]

    def __getattr__(self, name):
        """Returns attributes of the buffers"""
        return getattr(self._buffers, name)

    def __contains__(self, key):
        return key in self._buffers

    @abc.abstractmethod
    def add(self, data):
        raise NotImplementedError

    @abc.abstractmethod
    def extend(self, data):
        raise NotImplementedError


class FixedStorage(Storage):
    """
    This is a simple fixed storage buffer that supports direct allocation.

    It does not support adding additional data.
    """

    def __init__(self, buffers: Dict) -> None:
        assert "done" in buffers and buffers["done"].dtype == np.bool_, "'done' must be a np.bool_ in storage data."
        self._buffers = buffers
        # Set the necesary attributes
        self._buffers["done"][-1] = True
        self._capacity = len(self._buffers["done"])
        self._size = len(self._buffers["done"])
        (self._ends,) = np.where(self._buffers["done"])
        self._starts = np.concatenate(([0], self._ends[:-1] + 1))
        self._lengths = self._ends - self._starts + 1

    def add(self, data):
        raise ValueError("FixedStorage does not support adding")

    def extend(self, data):
        raise ValueError("FixedStorage does not support extending")


class NPQueue(object):
    def __init__(self, initial_capacity: int = 100, dtype=np.int64):
        self._arr = np.zeros(initial_capacity, dtype=dtype)
        self._start_idx = 0
        self._end_idx = 0

    def _reset(self):
        current_size = self._end_idx - self._start_idx
        new_arr = np.zeros(2 * current_size, dtype=self._arr.dtype)
        new_arr[:current_size] = self.view()
        self._arr = new_arr
        self._start_idx = 0
        self._end_idx = current_size

    def append(self, value):
        self._arr[self._end_idx] = value
        self._end_idx += 1
        if self._end_idx == len(self._arr):
            self._reset()

    def pop(self):
        self._end_idx -= 1
        return self._arr[self._end_idx + 1]

    def popleft(self):
        self._start_idx += 1

    def view(self):
        return self._arr[self._start_idx : self._end_idx]

    def __len__(self):
        return self._end_idx - self._start_idx

    def first(self):
        assert self._end_idx - self._start_idx > 0
        return self._arr[self._start_idx]

    def last(self):
        assert self._end_idx - self._start_idx > 0
        return self._arr[self._end_idx - 1]

    def __str__(self):
        return self.view().__str__()


class CircularStorage(Storage):
    """
    A ciruclar storage buffer with fixed capacities

    One tradeoff of this design is that trajectories can be split across the buffer
    However, this only happens to at most one.
    """

    def __init__(self, buffer_space: Union[Dict, gym.spaces.Dict], capacity: Optional[int] = None) -> None:
        assert isinstance(buffer_space, (dict, gym.spaces.Dict))
        self._buffers = utils.np_dataset_alloc(buffer_space, capacity)
        self._capacity = capacity
        self._size = 0
        self._idx = 0  # Note that unlike most replay buffers, this idx attribute is absolute

        # Just track the absolute end idx markers...
        self._ends_deque = NPQueue(initial_capacity=100, dtype=np.int64)
        self._ends = np.array([], dtype=np.int64)
        self._starts = np.array([], dtype=np.int64)
        self._lengths = np.array([], dtype=np.int64)
        self._skipped_last = True

    def _update_markers(self, new_ends: Iterable = ()):
        """
        This function updates _ends, _starts, and _lenghts
        It must be called everytime there are changes to the buffer.

        The last item of the deque is _always_ self._idx - 1, the last point of the last.
        The deque must always have something inserted at any number % self.capacity = self.capacity - 1
        """
        use_streaming_update = len(new_ends) > 0 and not self._skipped_last
        if self._skipped_last:
            self._skipped_last = False
        else:
            self._ends_deque.pop()

        for end in new_ends:
            self._ends_deque.append(end)  # Add the new ends
        if len(self._ends_deque) > 0 and self._idx - 1 == self._ends_deque.last():
            # Never double queue a value
            self._skipped_last = True
        else:
            self._ends_deque.append(self._idx - 1)  # Add back self._idx

        # Remove the old values
        current_floor = max(0, self._idx - self._capacity)
        while self._ends_deque.first() < current_floor:
            self._ends_deque.popleft()

        # Compute the new ends
        if use_streaming_update:  # len(new_ends) == 0:
            # The number of ends haven't changed. All we did was update self._idx
            # This means we just need to update the last part of the storage buffer
            self._ends[-1] = (self._idx - 1) % self._capacity
            self._lengths[-1] = self._ends[-1] - self._starts[-1] + 1
        else:
            # The number of ends changed... we need to recompute
            self._ends = self._ends_deque.view() % self._capacity
            self._starts = np.roll(self._ends + 1, 1) % self._size
            self._lengths = self._ends - self._starts + 1

    def add(self, data):
        assert not isinstance(data["done"], (list, np.ndarray)), "For adding lists use extend."
        assert isinstance(data["done"], (bool, np.bool_))

        buffer_idx = self._idx % self._capacity
        if buffer_idx + 1 == self._capacity:  # this is the last datapoint before wrap. Mark as done.
            data["done"] = True
        new_ends = [self._idx] if data["done"] else []
        utils.set_in_batch(self._buffers, data, buffer_idx, buffer_idx + 1)
        self._idx += 1
        self._size = min(self._idx, self._capacity)
        self._update_markers(new_ends=new_ends)

    def extend(self, data):
        """
        Adds multiple data points (assumes batching).
        """
        assert isinstance(data["done"], (list, np.ndarray)), "For adding elements use add."
        assert isinstance(data["done"][0], (bool, np.bool_))
        num_to_add = len(data["done"])
        buffer_idx = self._idx % self._capacity
        if buffer_idx + num_to_add > self._capacity:
            num_b4_wrap = self._capacity - buffer_idx  # How much empty space we have
            data["done"][num_b4_wrap - 1] = True  # Mark as true since it's going over the boundary
            self.extend(utils.get_from_batch(data, 0, num_b4_wrap))
            self.extend(utils.get_from_batch(data, num_b4_wrap, num_to_add))
        else:
            # compute the new absolute end_idxs
            (end_idxs,) = np.where(data["done"])
            end_idxs = self._idx + end_idxs
            if buffer_idx + num_to_add == self.capacity:
                data["done"][-1] = True
            utils.set_in_batch(self._buffers, data, buffer_idx, buffer_idx + num_to_add)
            self._idx += num_to_add
            self._size = min(self._idx, self._capacity)
            # Update the end buffers
            self._update_markers(new_ends=end_idxs.tolist())
