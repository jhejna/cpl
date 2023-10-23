import argparse
import io

import gym
import numpy as np

from research.datasets.replay_buffer import storage
from research.envs.metaworld import MetaWorldSawyerImageWrapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--output", type=str, required=True, help="Path to output the new dataset")
    parser.add_argument("--resolution", type=int, default=64, help="Resolution to render")
    parser.add_argument("--env", type=str, required=True)

    args = parser.parse_args()

    data = storage.load_data(args.path, exclude_keys=["mask"])

    assert "state" in data

    env = gym.make(args.env)
    env = MetaWorldSawyerImageWrapper(env, width=args.resolution, height=args.resolution)
    env.reset()  # Moves the camera

    # Allocate the whole buffer in advance to make sure we can fit it.
    num_segments, segment_length = data["obs"].shape[:2]
    images = np.zeros((num_segments, segment_length, 3, args.resolution, args.resolution), dtype=np.uint8)
    for segment in range(num_segments):
        # Render all of the images
        for t in range(segment_length):
            env.set_state(data["state"][segment, t])
            img = env._get_image()
            images[segment, t] = img

    # save the data
    data["obs"] = images

    with io.BytesIO() as bs:
        np.savez_compressed(bs, **data)
        bs.seek(0)
        with open(args.output, "wb") as f:
            f.write(bs.read())
