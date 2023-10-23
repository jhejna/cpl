import copy
import ctypes
import gc
import multiprocessing as mp
import queue
import sys
import time
from enum import Enum
from typing import Any, Callable, Optional

import gym
import numpy as np

from . import utils

"""
Env runners are used by the Trainer to run the environments.

They vary, and algorithms can take advantage of them to do special asynchronous work.
"""

__all__ = ["AsyncEnv", "MPRunner"]


class CloudpickleWrapper:
    """Wrapper that uses cloudpickle to pickle and unpickle the result."""

    def __init__(self, fn: Callable):
        """Cloudpickle wrapper for a function."""
        self.fn = fn

    def __getstate__(self):
        """Get the state using `cloudpickle.dumps(self.fn)`."""
        import cloudpickle

        return cloudpickle.dumps(self.fn)

    def __setstate__(self, ob):
        """Sets the state with obs."""
        import pickle

        self.fn = pickle.loads(ob)

    def __call__(self):
        """Calls the function `self.fn` with no arguments."""
        return self.fn()


def alloc_shared_buffer(space: Any):
    # Note that we use RAW Values here! there is no lock.
    # The programmer will need to explicitly handle taking turns reading / writing.
    if isinstance(space, (gym.spaces.Dict, dict)):
        return {k: alloc_shared_buffer(v) for k, v in space.items()}
    elif isinstance(space, (np.ndarray, gym.spaces.Box, gym.spaces.Discrete)):
        size = int(np.prod(space.shape) * space.dtype.itemsize)
        return mp.RawArray(ctypes.c_byte, size)
    else:
        raise ValueError("Unsupported type passed to `alloc_shared_mem`.")


def read_shared_buffer(shared_buffer: Any, space: gym.Space):
    if isinstance(space, (gym.spaces.Dict, dict)):
        return {k: read_shared_buffer(shared_buffer[k], v) for k, v in space.items()}
    elif isinstance(space, (gym.spaces.Box, gym.spaces.Discrete)):
        return np.frombuffer(shared_buffer, dtype=space.dtype).reshape(space.shape)
    else:
        raise ValueError("Encountered invalid location in")


def write_shared_buffer(shared_buffer: Any, space: gym.Space, value: Any):
    if isinstance(space, (gym.spaces.Dict, dict)):
        return {k: write_shared_buffer(shared_buffer[k], v, value[k]) for k, v in space.items()}
    elif isinstance(space, (gym.spaces.Box, gym.spaces.Discrete)):
        dest = np.frombuffer(shared_buffer, dtype=space.dtype)
        np.copyto(dest, np.asarray(value, dtype=space.dtype).flatten())
    else:
        raise ValueError("Encountered invalid space in `write_shared_buffer`")


class AsyncState(Enum):
    DEFAULT = "default"
    WAITING_RESET = "reset"
    WAITING_STEP = "step"


class AsyncEnv(gym.Env):
    """
    A container for an environment that will be run completely detached in a subprocess.
    """

    def __init__(
        self, env_fn: Callable, observation_space: Optional[gym.Space] = None, action_space: Optional[gym.Space] = None
    ):
        if observation_space is None or action_space is None:
            # Fetch the spaces and then delete
            dummy_env = env_fn()
            self.observation_space = utils.space_copy(dummy_env.observation_space)
            self.action_space = utils.space_copy(dummy_env.action_space)
            dummy_env.close()
            del dummy_env
            gc.collect()
        else:
            self.observation_space = observation_space
            self.action_space = action_space

        # NOTE: We use the default context for simplicity. Later we could look into custom contexts.
        # See: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/vector/async_vector_env.py

        obs_buffer = alloc_shared_buffer(self.observation_space)
        self.observation = read_shared_buffer(obs_buffer, self.observation_space)  # Gets mem locations
        self.action_buffer = alloc_shared_buffer(self.action_space)
        self.parent_pipe, child_pipe = mp.Pipe()
        self.process = mp.Process(
            target=_async_env_worker,
            name="AsyncEnvWorker",
            args=(CloudpickleWrapper(env_fn), child_pipe, self.parent_pipe, obs_buffer, self.action_buffer),
        )
        self.process.start()
        child_pipe.close()

        self.state = AsyncState.DEFAULT

    def step_send(self, action):
        assert self.state == AsyncState.DEFAULT, "Trying to step while not in AsycState.DEFAULT"
        # Write the action
        write_shared_buffer(self.action_buffer, self.action_space, action)
        self.parent_pipe.send("step")
        self.state = AsyncState.WAITING_STEP

    def step_recv(self):
        assert self.state == AsyncState.WAITING_STEP, "Called reset_recv when state was not AsyncState.WAITING_RESET"
        success, data = self.parent_pipe.recv()
        assert success, "Pipe communication failed."
        reward, done, discount = data
        self.state = AsyncState.DEFAULT
        return copy.deepcopy(self.observation), reward, done, dict(discount=discount)

    def step(self, action):
        self.step_send(action)
        return self.step_recv()

    def reset_send(self):
        assert self.state == AsyncState.DEFAULT, "Trying to reset while not in AsycState.DEFAULT"
        self.parent_pipe.send("reset")
        self.state = AsyncState.WAITING_RESET

    def reset_recv(self):
        assert self.state == AsyncState.WAITING_RESET, "Called reset_recv when state was not AsyncState.WAITING_RESET"
        success, data = self.parent_pipe.recv()
        assert success, "Pipe communication failed."
        self.state = AsyncState.DEFAULT
        return copy.deepcopy(self.observation)

    def reset(self):
        self.reset_send()
        return self.reset_recv()

    def close(self):
        self.parent_pipe.send("close")
        time.sleep(2)
        if self.process.is_alive():
            self.process.terminate()


def _async_env_worker(env_fn, pipe, parent_pipe, obs_buffer, action_buffer):
    env = env_fn()
    parent_pipe.close()  # Close that end of the pipe.
    action = read_shared_buffer(action_buffer, env.action_space)  # persistent array -- modified by other process!
    ep_length = 0
    try:
        while True:
            command = pipe.recv()
            if command == "reset":
                obs = env.reset()
                ep_length = 0
                write_shared_buffer(obs_buffer, env.observation_space, obs)
                pipe.send((True, None))
            elif command == "step":
                obs, reward, done, info = env.step(action)
                ep_length += 1
                # Compute the discount if needed.
                if "discount" in info:
                    discount = info["discount"]
                elif hasattr(env, "_max_episode_steps") and ep_length == env._max_episode_steps:
                    discount = 1.0
                else:
                    discount = 1 - float(done)
                write_shared_buffer(obs_buffer, env.observation_space, obs)
                pipe.send((True, (reward, done, discount)))
            elif command == "close":
                break
            else:
                pipe.send((False, None))
                raise ValueError("Invalid command sent to subprocess worker.")

    except (KeyboardInterrupt, Exception):
        pipe.send((False, None))
        print(sys.exec_info()[:2])
    finally:
        env.close()


class MPRunner(object):
    """
    A simple class that creates a subprocess to run a function for an environment
    It is given the environment function and the function to call.
    """

    def __init__(
        self,
        env_fn,
        fn: Optional[Callable] = None,
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        **kwargs,
    ):
        self.env_fn = env_fn
        self.kwargs = kwargs
        self.fn = fn
        self._started = False

    def start(self, fn: Optional[Callable] = None, **kwargs):
        assert not self._started, "Cannot start MPRunner Twice!"
        self._started = True
        # Construct a metrics queue...
        if fn is not None:
            self.fn = fn
        fn_kwargs = copy.deepcopy(self.kwargs)
        fn_kwargs.update(kwargs)
        # Start the MP context
        self._queue = mp.Queue()
        self.process = mp.Process(
            target=self.fn, name="MPRunner", args=(CloudpickleWrapper(self.env_fn), self._queue), kwargs=fn_kwargs
        )
        self.process.start()

    @property
    def started(self):
        return self._started

    def __call__(self, block=False):
        metrics = {}
        try:
            while True:
                # Try until we except! (Could be the first time, or the last time)
                # Then return the most recent eval metrics found.
                metrics = self._queue.get(block=block, timeout=None)
                assert isinstance(
                    metrics, dict
                ), "MPRunner subprocess did not return a metrics dict. It may have failed."
        except queue.Empty:
            return metrics

    def step(self, *args, **kwargs):
        raise ValueError("Using Async Runner! ")

    def reset(self, *args, **kwargs):
        raise ValueError("Use step_send and step_recv!")

    def close(self):
        if self._started and self.process.is_alive():
            self.process.terminate()
