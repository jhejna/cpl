import argparse
import collections
import io
import os

import numpy as np

from research.datasets import ReplayBuffer
from research.utils import utils
from research.utils.config import Config

if __name__ == "__main__":
    # This is a short script that generates a pairwise preference dataset from a ReplayBuffer

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to the ReplayBuffer")
    parser.add_argument("--output", type=str, required=True, help="Output path for the dataset")
    parser.add_argument("--size", type=int, default=20000, help="How many data points to sample")
    parser.add_argument("--segment-size", type=int, default=64, help="How large to make segments")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to oracle model")

    args = parser.parse_args()

    # Get the model
    config_path = os.path.dirname(args.checkpoint) if args.checkpoint.endswith(".pt") else args.checkpoint
    config = Config.load(config_path)
    config["checkpoint"] = None  # Set checkpoint to None, we don't actually need to load it.
    config = config.parse()
    env_fn = config.get_train_env_fn()
    if env_fn is None:
        env_fn = config.get_eval_env_fn()
    env = env_fn()

    # Load the data
    assert os.path.exists(args.path)
    replay_buffer = ReplayBuffer(
        env.observation_space, env.action_space, distributed=False, path=args.path, sample_fn="sample_sequence"
    )

    data = []
    scores = collections.defaultdict(list)
    batch_size = 256
    remaining_data_points = args.size
    # sample num segments...
    while remaining_data_points > 0:
        sample_size = min(batch_size, remaining_data_points)
        batch = replay_buffer.sample(
            batch_size=sample_size,
            sample_by_timesteps=True,
            seq_length=args.segment_size,
            pad=0,
            seq_keys=("obs", "action", "reward", "state", "timestep"),
        )
        del batch["mask"]
        data.append(batch)
        remaining_data_points -= sample_size

    assert remaining_data_points == 0, "Must have zero remaining segments"

    data = utils.concatenate(*data, dim=0)
    data = utils.remove_float64(data)

    data_length = data["action"].shape[0]
    assert all([len(v) == data_length for v in data.values()])

    # Save the feedback data to a path.
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **data)
        bs.seek(0)
        with open(args.output, "wb") as f:
            f.write(bs.read())
