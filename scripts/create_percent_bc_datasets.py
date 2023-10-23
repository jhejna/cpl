import argparse

import gym
import numpy as np

from research.datasets import ReplayBuffer
from research.utils import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--percent", type=float, default=0.05)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    env = gym.make(args.env)

    replay_buffer = ReplayBuffer(
        env.observation_space, env.action_space, path=args.path, distributed=False, capacity=None
    )  # Load all of the data.

    storage = replay_buffer._storage

    # Compute all of the rewards values
    ep_rewards = []
    for i in range(len(storage.starts)):
        start, end = storage.starts[i], storage.ends[i]
        ep_rewards.append(storage["reward"][start:end].sum())

    # Compute the percentile
    ep_rewards = np.array(ep_rewards)
    # argsort
    ep_idxs = np.argsort(ep_rewards)
    ep_idxs = ep_idxs[::-1]  # Reverse the order
    ep_idxs = ep_idxs[: int(len(ep_idxs) * args.percent)]
    print(len(ep_rewards))
    print(ep_rewards[ep_idxs])

    # Construct a new replay buffer that is just the top 10% of episodes
    new_replay_buffer = ReplayBuffer(
        env.observation_space, env.action_space, path=args.path, distributed=False, capacity=storage.size
    )

    for _idx in ep_idxs:
        start, end = storage.starts[i], storage.ends[i]
        ep = {}
        for k in storage.keys():
            ep[k] = utils.get_from_batch(storage[k], start, end)
        new_replay_buffer.extend(**ep)

    # save the new replay buffer
    new_replay_buffer.save(args.output)

    # Done!
