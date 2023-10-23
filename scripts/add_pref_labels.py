import argparse
import collections
import io
import math
import os

import numpy as np
import torch

from research.datasets.replay_buffer import storage
from research.utils import utils
from research.utils.config import Config

if __name__ == "__main__":
    # This is a short script that generates a pairwise preference dataset from a ReplayBuffer
    # It does so using multiple possible different metrics.

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to the Dataset")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to oracle model")
    parser.add_argument("--prefix", type=str, default="rl", required=False, help="label_prefix")
    parser.add_argument("--discount", type=float, default=0.99, help="Path to oracle model")
    parser.add_argument("--samples", type=int, default=64, help="Number of MCMC samples for the policy.")
    parser.add_argument("--segment-length", type=int, default=None, help="Number of MCMC samples for the policy.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=64)

    args = parser.parse_args()

    # Get the model
    assert args.checkpoint.endswith(".pt"), "Must provide a model checkpoint"
    config = Config.load(os.path.dirname(args.checkpoint))
    config["checkpoint"] = None  # Set checkpoint to None, we don't actually need to load it.
    config = config.parse()
    env_fn = config.get_train_env_fn()
    if env_fn is None:
        env_fn = config.get_eval_env_fn()
    env = env_fn()

    model = config.get_model(observation_space=env.observation_space, action_space=env.action_space, device=args.device)
    model.load(args.checkpoint)

    # Load the data
    dataset = storage.load_data(args.path, exclude_keys=["mask"])
    # Dataset size
    dataset_size = dataset["action"].shape[0]
    segment_length = dataset["action"].shape[1]

    if args.segment_length < segment_length:
        start_idx = np.random.randint(low=0, high=segment_length - args.segment_length, size=(dataset_size))
        new_dataset = {
            k: np.empty_like(v[:, : args.segment_length]) for k, v in dataset.items() if len(v.shape) > 1
        }  # Remove old labels
        for i in range(dataset_size):
            for k in new_dataset.keys():
                new_dataset[k][i] = dataset[k][i][start_idx[i] : start_idx[i] + args.segment_length]

        dataset = new_dataset

    metrics = collections.defaultdict(list)
    for i in range(math.ceil(dataset_size / args.batch_size)):  # Need to use ceil to get all data points.
        idxs = np.arange(i * args.batch_size, min((i + 1) * args.batch_size, dataset_size))

        # Predict the adv metrics
        with torch.no_grad():
            obs = torch.from_numpy(dataset["obs"][idxs]).to(model.device)  # (B, S, D)
            action = torch.from_numpy(dataset["action"][idxs]).to(model.device)  # (B, S, D)
            reward = torch.from_numpy(dataset["reward"][idxs]).to(model.device)

            obs = model.network.encoder(obs)
            dist = model.network.actor(obs)  # (B, S, D_a)

            if isinstance(dist, torch.distributions.Distribution):
                lp = dist.log_prob(torch.clamp(action, min=-0.999, max=0.999)).sum(dim=-1)  # (B,)
                l2 = torch.square(dist.base_dist.loc - action).sum(dim=-1).sum(dim=-1)  # (B,)
                metrics[args.prefix + "_lp"].append(lp.cpu().numpy())
                metrics[args.prefix + "_l2"].append(l2.cpu().numpy())
            else:
                l2 = torch.square(dist - action).sum(dim=-1).sum(dim=-1)  # (B,)
                metrics[args.prefix + "_l2"].append(l2.cpu().numpy())

            if hasattr(model.network, "critic"):
                q = model.network.critic(obs, action).mean(dim=0)  # Avg over ensemble

                # Expand the observation to sample the value according to the policy.
                obs = obs.unsqueeze(2).expand(-1, -1, args.samples, -1)  # (B, S, M, D)
                dist = model.network.actor(obs)  # (B, S, M, D_a)
                sampled_action = dist.sample()  # (B, S, M, D_a)
                v = model.network.critic(obs, sampled_action).mean(dim=0)  # (B, S, M)
                # Maybe add log prob??? would be v = v - dist.log_prob(sampled_action)
                std = torch.std(v, dim=2).mean()
                v = v.mean(dim=2)  # (B, S)

                discount = torch.pow(
                    args.discount * torch.ones(q.shape[1], device=q.device, dtype=torch.float),
                    torch.arange(q.shape[1], device=q.device),
                )
                discount = discount.unsqueeze(0)

                direct_adv = (q - v).sum(dim=-1)
                discounted_direct_adv = (discount * (q - v)).sum(dim=-1)
                sum_adv = reward[:, :-1].sum(dim=-1) + v[:, -1] - v[:, 0]
                discounted_sum_adv = (
                    (discount * reward)[:, :-1].sum() + (args.discount ** reward.shape[1]) * v[:, -1] - v[:, 0]
                )

                metrics[args.prefix + "_dir"].append(direct_adv.cpu().numpy())
                metrics[args.prefix + "_dis_dir"].append(discounted_direct_adv.cpu().numpy())
                metrics[args.prefix + "_sum"].append(sum_adv.cpu().numpy())
                metrics[args.prefix + "_dis_sum"].append(discounted_sum_adv.cpu().numpy())

    metrics = {k: utils.concatenate(*v, dim=0) for k, v in metrics.items()}
    metrics = utils.remove_float64(metrics)

    dataset.update(metrics)  # Update the dataset to containe the new metrics

    assert all([len(v) == dataset_size for v in dataset.values()])

    # Save the feedback data to a path.
    output_path = args.path
    if args.segment_length != segment_length:
        # Split the path and then rejoin.
        path, ext = os.path.splitext(output_path)
        output_path = path + "_s" + str(args.segment_length) + ext
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **dataset)
        bs.seek(0)
        with open(output_path, "wb") as f:
            f.write(bs.read())
