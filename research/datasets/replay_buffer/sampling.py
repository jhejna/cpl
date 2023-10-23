import copy
from typing import Callable, Optional, Tuple

import numpy as np

from research.utils import utils

from .storage import Storage

"""
This file defines a number of sampling functions used by the replay buffer.

Each sample function returns tensors of the following shape:
(Batch, Time, dims...)
and requires `storage` and `discount` arguments.

Many of these functions have large blocks of repeated code, but
are implemented separately for readability and performance optimiztaion.

Sequences are sampled as follows:
-stack_length ...  -1, 0, 1, 2, ..., seq_length
|         stack      |idx|        seq          |
The stack parameter will always be sampled immediately, and is desinged to be used as context
to the network.
Stack will not obey nstep returns. (negative indexing)

Everything is sampled in batches directly from memory (preferred)
If batch_size is set to one, then a squeeze operation will be performed at the very end.

Samples are returned as with shape: (Batch, Time, Dims...)
if seq or stack dims are set to 1, then these parameters are ignored.
"""


def _get_ep_idxs(storage: Storage, batch_size: int = 1, sample_by_timesteps: bool = True, min_length: int = 2):
    if batch_size is None or batch_size > 1:
        ep_idxs = np.arange(len(storage.lengths))[storage.lengths >= min_length]
        if sample_by_timesteps:
            # Lower the lengths by the min_length - 1 to give the number of valid sequences.
            lengths = storage.lengths[ep_idxs] - (min_length - 1)
            p = lengths / lengths.sum()
            ep_idxs = np.random.choice(ep_idxs, size=(batch_size,), replace=True, p=p)
        else:
            ep_idxs = ep_idxs[np.random.randint(0, len(ep_idxs), size=(batch_size,))]
        return ep_idxs
    else:
        # Use a different, much faster sampling scheme for batch_size = 1
        assert sample_by_timesteps is False, "Cannot sample by timesteps with batch_size=1, it's too slow!"
        ep_idx = np.random.randint(0, len(storage.lengths))
        if storage.lengths[ep_idx] < min_length:
            return _get_ep_idxs(storage, batch_size, sample_by_timesteps, min_length)
        else:
            return np.array([ep_idx], np.int64)


def sample(
    storage: Storage,
    batch_size: int = 1,
    sample_by_timesteps: bool = True,
    stack: int = 1,
    stack_keys: Tuple = (),
    discount: float = 0.99,
):
    """
    Default sampling for imitation learning.
    Returns (obs, action, ... keys) batches.
    """
    min_length = 2
    ep_idxs = _get_ep_idxs(storage, batch_size, sample_by_timesteps, min_length)
    # sample position within the episode randomly
    # Note that there is a plus one offset here to account for the difference
    # between the obs and action position
    offsets = np.random.randint(1, storage.lengths[ep_idxs])

    if stack > 1:
        assert len(stack_keys) > 1, "Provided stack > 1 but no stack keys"
        stack_offsets = np.expand_dims(offsets, axis=-1) + np.arange(-stack + 1, 1)
        stack_offsets = np.clip(stack_offsets, 0, None)  # Clip to zero as lowest offset = start of episode
        stack_idxs = np.expand_dims(storage.starts[ep_idxs], axis=-1) + stack_offsets
    idxs = storage.starts[ep_idxs] + offsets

    # Sample from the dataset
    batch = {}
    for k in storage.keys():
        sample_idxs = stack_idxs if k in stack_keys else idxs
        if k == "obs":
            sample_idxs = sample_idxs - 1
        if k == "discount":
            batch[k] = discount * utils.get_from_batch(storage[k], sample_idxs)
        else:
            batch[k] = utils.get_from_batch(storage[k], sample_idxs)
    return batch


def sample_qlearning(
    storage: Storage,
    batch_size: int = 1,
    sample_by_timesteps: bool = True,
    nstep: int = 1,
    stack: int = 1,
    stack_keys: Tuple = (),
    discount: float = 0.99,
):
    """
    Default sampling for reinforcement learning.
    Returns (obs, action, reward, discount, next_obs) batches.

    Similar to the default `sample` method, but includes limits sampling
    to the required keys.
    """

    min_length = 1 + nstep
    ep_idxs = _get_ep_idxs(storage, batch_size, sample_by_timesteps, min_length)
    # sample position within the episode randomly
    # There is a plus one offset here to account for the difference between the obs and action position
    # length = start - end + 1 ie 0 to 2 has length of 3 bc all indexes are filled.
    # There is a nstep - 1 to account for the fact that we must be able to sample next_obs in the future.
    offsets = np.random.randint(1, storage.lengths[ep_idxs] - nstep + 1)

    if stack > 1:
        assert len(stack_keys) > 1, "Provided stack > 1 but no stack keys"
        stack_offsets = np.expand_dims(offsets, axis=-1) + np.arange(-stack + 1, 1)
        stack_offsets = np.clip(stack_offsets, 0, None)  # Clip to zero as lowest offset = start of episode
        stack_idxs = np.expand_dims(storage.starts[ep_idxs], axis=-1) + stack_offsets
    idxs = storage.starts[ep_idxs] + offsets

    # Get the observation indexes
    obs_idxs = stack_idxs if "obs" in stack_keys else idxs
    next_obs_idxs = obs_idxs + nstep - 1
    obs_idxs = obs_idxs - 1

    # Get the action indexes
    action_idxs = stack_idxs if "action" in stack_keys else idxs

    obs = utils.get_from_batch(storage["obs"], obs_idxs)
    action = utils.get_from_batch(storage["action"], action_idxs)
    reward = np.zeros_like(storage["reward"][idxs])
    discount_batch = np.ones_like(storage["discount"][idxs])
    for i in range(nstep):
        reward += discount_batch * storage["reward"][idxs + i]
        discount_batch *= discount * storage["discount"][idxs + i]
    next_obs = utils.get_from_batch(storage["obs"], next_obs_idxs)

    return dict(obs=obs, action=action, reward=reward, discount=discount_batch, next_obs=next_obs)


def sample_sequence(
    storage: Storage,
    batch_size: int = 1,
    sample_by_timesteps: bool = True,
    seq_length: int = 1,
    seq_keys: Tuple = (),
    pad: int = 0,
    discount: float = 0.99,
):
    """
    Sequence sampling for imitation learning.
    Returns a batch of (obs, action, ... sequences)
    """
    assert pad < seq_length, "Cannot use seq length equal to pad."
    min_length = seq_length - pad + 1
    ep_idxs = _get_ep_idxs(storage, batch_size, sample_by_timesteps, min_length)
    starts, ends = storage.starts[ep_idxs], storage.ends[ep_idxs]
    # Add 2: one for not-inclusive, one for seq_len offset.
    idxs = np.random.randint(starts + 1, ends + (2 + pad - seq_length))
    batch = {}

    # After the base indexes, determine if we have other idxs
    if seq_length > 1:
        assert len(seq_keys) > 1, "Provided seq_length > 1 but no seq keys"
        # nstep to seq
        seq_idxs = np.expand_dims(idxs, axis=-1) + np.arange(seq_length)  # (B, T)
        # Compute the mask
        mask = seq_idxs > np.expand_dims(ends, axis=-1)
        # Trim down to save mem usage by returning a view of the same data.
        seq_idxs = np.minimum(seq_idxs, np.expand_dims(ends, axis=-1))
        batch["mask"] = mask

    # Sample from the dataset
    for k in storage.keys():
        sample_idxs = seq_idxs if k in seq_keys else idxs
        if k == "obs":
            sample_idxs = sample_idxs - 1
        if k == "discount":
            batch[k] = discount * utils.get_from_batch(storage[k], sample_idxs)
        else:
            batch[k] = utils.get_from_batch(storage[k], sample_idxs)

    return batch


def sample_her(
    storage: Storage,
    batch_size: int = 1,
    sample_by_timesteps: bool = True,
    achieved_key: str = "achieved_goal",
    goal_key: str = "desired_goal",
    stack: int = 1,
    stack_keys: Tuple = (),
    strategy: str = "future",
    relabel_fraction: float = 0.5,
):
    """
    Default sampling for imitation learning.
    Returns (obs, action, ... keys) batches.
    """
    assert isinstance(storage["obs"], dict)

    min_length = 2
    ep_idxs = _get_ep_idxs(storage, batch_size, sample_by_timesteps, min_length)

    # sample position within the episode randomly
    # Note that there is a plus one offset here to account for the difference
    # between the obs and action position
    offsets = np.random.randint(1, storage.lengths[ep_idxs])

    if stack > 1:
        assert len(stack_keys) > 1, "Provided stack > 1 but no stack keys"
        stack_offsets = np.expand_dims(offsets, axis=-1) + np.arange(-stack + 1, 1)
        stack_offsets = np.clip(stack_offsets, 0, None)  # Clip to zero as lowest offset = start of episode
        stack_idxs = np.expand_dims(storage.starts[ep_idxs], axis=-1) + stack_offsets
    idxs = storage.starts[ep_idxs] + offsets

    her_idxs = np.where(np.random.uniform(size=idxs.shape) < relabel_fraction)
    if strategy == "last":
        goal_idxs = storage.ends[ep_idxs[her_idxs]]
    elif strategy == "next":
        goal_idxs = idxs[her_idxs]
    elif strategy == "future":
        goal_idxs = np.random.randint(idxs[her_idxs], storage.ends[ep_idxs[her_idxs]] + 1)
    else:
        raise ValueError("Invalid HER strategy chosen.")

    if relabel_fraction < 1.0:
        # Need to copy out existing values.
        desired = copy.deepcopy(utils.get_from_batch(storage["obs"][goal_key], idxs - 1))
        achieved = utils.get_from_batch(storage["obs"][achieved_key], goal_idxs)
        utils.set_in_batch(desired, achieved, her_idxs)
        horizon = (
            utils.get_from_batch(storage["horizon"], idxs - 1)
            if "horizon" in storage
            else -100 * np.ones_like(idxs, dtype=np.int)
        )
        horizon[her_idxs] = goal_idxs - idxs[her_idxs] + 1
    else:
        # Grab directly from the buffer.
        desired = utils.get_from_batch(storage["obs"][achieved_key], goal_idxs)
        horizon = goal_idxs - idxs + 1  # Horizon = goal_idxs - obs_idxs

    if stack > 1 and "obs" in stack_keys:
        # Add temporal dimension if we stack the achieved frames.
        desired = np.expand_dims(desired, axis=1)  # Shape (B, 1, dims...)

    # Sample from the dataset
    batch = {}
    for k in storage.keys():
        sample_idxs = stack_idxs if k in stack_keys else idxs
        if k == "obs":
            # Ignore sampling the goal key
            batch[k] = {
                obs_key: utils.get_from_batch(storage[k][obs_key], sample_idxs - 1)
                for obs_key in storage[k].keys()
                if obs_key != goal_key
            }
        else:
            batch[k] = utils.get_from_batch(storage[k], sample_idxs)

    # Update the batch to use the newly set desired goals
    batch["obs"][goal_key] = desired
    batch["horizon"] = horizon

    return batch


def sample_her_qlearning(
    storage: Storage,
    batch_size: int = 1,
    sample_by_timesteps: bool = True,
    nstep: int = 1,
    achieved_key: str = "achieved_goal",
    goal_key: str = "desired_goal",
    stack: int = 1,
    stack_keys: Tuple = (),
    strategy: str = "future",
    relabel_fraction: float = 0.5,
    reward_fn: Optional[Callable] = None,
    discount=0.99,
):
    """
    Default sampling for reinforcement learning with HER
    Returns (obs, action, reward, discount, next_obs) batches.

    Similar to the default `sample` method, but includes limits sampling
    to the required keys.
    """
    min_length = 1 + nstep
    ep_idxs = _get_ep_idxs(storage, batch_size, sample_by_timesteps, min_length)

    # sample position within the episode randomly
    # There is a plus one offset here to account for the difference between the obs and action position
    # length = start - end + 1 ie 0 to 2 has length of 3 bc all indexes are filled.
    # There is a nstep - 1 to account for the fact that we must be able to sample next_obs in the future.
    offsets = np.random.randint(1, storage.lengths[ep_idxs] - nstep + 1)

    if stack > 1:
        assert len(stack_keys) > 1, "Provided stack > 1 but no stack keys"
        stack_offsets = np.expand_dims(offsets, axis=-1) + np.arange(-stack + 1, 1)
        stack_offsets = np.clip(stack_offsets, 0, None)  # Clip to zero as lowest offset = start of episode
        stack_idxs = np.expand_dims(storage.starts[ep_idxs], axis=-1) + stack_offsets
    idxs = storage.starts[ep_idxs] + offsets

    her_idxs = np.where(np.random.uniform(size=idxs.shape) < relabel_fraction)
    if strategy == "last":
        goal_idxs = storage.ends[ep_idxs[her_idxs]]
    elif strategy == "next":
        goal_idxs = idxs[her_idxs]
    elif strategy == "future":
        goal_idxs = np.random.randint(idxs[her_idxs], storage.ends[ep_idxs[her_idxs]] + 1)
    else:
        raise ValueError("Invalid HER strategy chosen.")

    if relabel_fraction < 1.0:
        # Need to copy out existing values.
        desired = copy.deepcopy(utils.get_from_batch(storage["obs"][goal_key], idxs - 1))
        achieved = utils.get_from_batch(storage["obs"][achieved_key], goal_idxs)
        utils.set_in_batch(desired, achieved, her_idxs)
        horizon = (
            utils.get_from_batch(storage["horizon"], idxs - 1)
            if "horizon" in storage
            else -100 * np.ones_like(idxs, dtype=np.int)
        )
        horizon[her_idxs] = np.ceil((goal_idxs - idxs[her_idxs] + 1) / nstep).astype(np.int64)
    else:
        # Grab directly from the buffer.
        desired = utils.get_from_batch(storage["obs"][achieved_key], goal_idxs)
        horizon = np.ceil((goal_idxs - idxs + 1) / nstep).astype(np.int64)  # Horizon = goal_idxs - obs_idxs

    # Get the observation indexes
    obs_idxs = stack_idxs if "obs" in stack_keys else idxs
    next_obs_idxs = obs_idxs + nstep - 1
    obs_idxs = obs_idxs - 1
    # Get the action indexes
    action_idxs = stack_idxs if "action" in stack_keys else idxs

    reward = np.zeros_like(storage["reward"][idxs])
    discount_batch = np.ones_like(storage["discount"][idxs])
    for i in range(nstep):
        if reward_fn is None:
            # If reward function is None, use sparse indicator reward.
            # If the next obs is the goal (aka horizon = 1), then reward = 1!
            step_reward = (horizon == i + 1).astype(np.float32)
        else:
            achieved = utils.get_from_batch(storage["obs"][achieved_key], idxs + i)
            step_reward = reward_fn(achieved, desired)
        reward += discount_batch * step_reward
        discount_batch *= discount * storage["discount"][idxs + i]

    if stack > 1 and "obs" in stack_keys:
        # Add temporal dimension if we stack the achieved frames.
        desired = np.expand_dims(desired, axis=1)  # Shape (B, 1, dims...)

    obs = {k: utils.get_from_batch(storage["obs"][k], obs_idxs) for k in storage["obs"].keys() if k != goal_key}
    obs[goal_key] = desired
    next_obs = {
        k: utils.get_from_batch(storage["obs"][k], next_obs_idxs) for k in storage["obs"].keys() if k != goal_key
    }
    next_obs[goal_key] = desired
    action = utils.get_from_batch(storage["action"], action_idxs)

    return dict(obs=obs, action=action, reward=reward, discount=discount_batch, next_obs=next_obs, horizon=horizon)


def sample_her_sequence(
    storage,
    batch_size=1,
    sample_by_timesteps=True,
    achieved_key="achieved_goal",
    goal_key="desired_goal",
    seq_length=1,
    seq_keys=(),
    pad=0,
    strategy="future",
    relabel_fraction=0.5,
):
    """
    Sequence sampling for imitation learning with HER.
    Returns a batch of (obs, action, ... sequences)
    """
    assert pad < seq_length, "Cannot use seq length equal to pad."
    min_length = seq_length - pad + 1
    ep_idxs = _get_ep_idxs(storage, batch_size, sample_by_timesteps, min_length)

    starts, ends = storage.starts[ep_idxs], storage.ends[ep_idxs]
    # Add 2: one for not-inclusive, one for seq_len offset.
    idxs = np.random.randint(starts + 1, ends + (2 + pad - seq_length))

    her_idxs = np.where(np.random.uniform(size=idxs.shape) < relabel_fraction)

    batch = {}

    # After the base indexes, determine if we have other idxs
    if seq_length > 1:
        assert len(seq_keys) > 1, "Provided seq_length > 1 but no seq keys"
        # nstep to seq
        seq_idxs = np.expand_dims(idxs, axis=-1) + np.arange(seq_length)  # (B, T)
        # Compute the mask
        mask = seq_idxs > np.expand_dims(ends, axis=-1)
        batch["mask"] = mask
        # Trim down to save mem usage by returning a view of the same data.
        seq_idxs = np.minimum(seq_idxs, np.expand_dims(ends, axis=-1))
        last_idxs = seq_idxs[..., -1]  # Get the last index from every sequence
    else:
        last_idxs = idxs  # only a single transition

    # NOTE: strategies must be re-worked for sequences!!!
    if strategy == "last":
        goal_idxs = storage.ends[ep_idxs[her_idxs]]
    elif strategy == "next":
        goal_idxs = last_idxs[her_idxs]  # Get the last position of the sequence
    elif strategy == "future":
        # Get a position between the end of the sequence (inclusive) and the end
        goal_idxs = np.random.randint(last_idxs[her_idxs], storage.ends[ep_idxs[her_idxs]] + 1)
    else:
        raise ValueError("Invalid HER strategy chosen.")

    if relabel_fraction < 1.0:
        # Need to copy out existing values.
        desired = copy.deepcopy(utils.get_from_batch(storage["obs"][goal_key], idxs - 1))
        achieved = utils.get_from_batch(storage["obs"][achieved_key], goal_idxs)
        utils.set_in_batch(desired, achieved, her_idxs)
        horizon = (
            utils.get_from_batch(storage["horizon"], idxs - 1)
            if "horizon" in storage
            else -100 * np.ones_like(idxs, dtype=np.int)
        )
        horizon[her_idxs] = goal_idxs - idxs[her_idxs] + 1
    else:
        # Grab directly from the buffer.
        desired = utils.get_from_batch(storage["obs"][achieved_key], goal_idxs)
        horizon = goal_idxs - idxs + 1  # Horizon = goal_idxs - obs_idxs

    if seq_length > 1 and "obs" in seq_keys:
        # Add temporal dimension if we stack the achieved frames.
        desired = np.expand_dims(desired, axis=1)  # Shape (B, 1, dims...)

    # Sample from the dataset
    for k in storage.keys():
        sample_idxs = seq_idxs if k in seq_keys else idxs
        if k == "obs":
            # Ignore sampling the goal key
            batch[k] = {
                obs_key: utils.get_from_batch(storage[k][obs_key], sample_idxs - 1)
                for obs_key in storage[k].keys()
                if obs_key != goal_key
            }
        else:
            batch[k] = utils.get_from_batch(storage[k], sample_idxs)

    return batch
