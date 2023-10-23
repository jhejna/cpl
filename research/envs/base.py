import gym
import numpy as np


def _get_space(low=None, high=None, shape=None, dtype=None):
    all_vars = [low, high, shape, dtype]
    if any([isinstance(v, dict) for v in all_vars]):
        all_keys = set()  # get all the keys
        for v in all_vars:
            if isinstance(v, dict):
                all_keys.update(v.keys())
        # Construct all the sets
        spaces = {}
        for k in all_keys:
            space_low = low.get(k, None) if isinstance(low, dict) else low
            space_high = high.get(k, None) if isinstance(high, dict) else high
            space_shape = shape.get(k, None) if isinstance(shape, dict) else shape
            space_type = dtype.get(k, None) if isinstance(dtype, dict) else dtype
            spaces[k] = _get_space(space_low, space_high, space_shape, space_type)
        # Construct the gym dict space
        return gym.spaces.Dict(**spaces)

    if shape is None and isinstance(high, int):
        assert low is None, "Tried to specify a discrete space with both high and low."
        return gym.spaces.Discrete(high)

    # Otherwise assume its a box.
    if low is None:
        low = -np.inf
    if high is None:
        high = np.inf
    if dtype is None:
        dtype = np.float32
    return gym.spaces.Box(low=low, high=high, shape=shape, dtype=dtype)


class EmptyEnv(gym.Env):

    """
    An empty holder for defining supervised learning problems
    It works by specifying the ranges and shapes.
    """

    def __init__(
        self,
        observation_low=None,
        observation_high=None,
        observation_shape=None,
        observation_dtype=np.float32,
        observation_space=None,
        action_low=None,
        action_high=None,
        action_shape=None,
        action_dtype=np.float32,
        action_space=None,
    ):
        if observation_space is not None:
            self.observation_space = observation_space
        else:
            self.observation_space = _get_space(observation_low, observation_high, observation_shape, observation_dtype)
        if action_space is not None:
            self.action_space = action_space
        else:
            self.action_space = _get_space(action_low, action_high, action_shape, action_dtype)

    def step(self, action):
        raise NotImplementedError("Empty Env does not have step")

    def reset(self, **kwargs):
        raise NotImplementedError("Empty Env does not have reset")
