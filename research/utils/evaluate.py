import collections
import os
from typing import Any, Dict, List, Optional

import gym
import imageio
import numpy as np
import torch

from . import utils

MAX_METRICS = {"success", "is_success", "completions"}
LAST_METRICS = {"goal_distance"}
MEAN_METRICS = {"discount"}
EXCLUDE_METRICS = {"TimeLimit.truncated"}


class EvalMetricTracker(object):
    """
    A simple class to make keeping track of eval metrics easy.
    Usage:
        Call reset before each episode starts
        Call step after each environment step
        call export to get the final metrics
    """

    def __init__(self):
        self.metrics = collections.defaultdict(list)
        self.ep_length = 0
        self.ep_reward = 0
        self.ep_metrics = collections.defaultdict(list)

    def reset(self) -> None:
        if self.ep_length > 0:
            # Add the episode to overall metrics
            self.metrics["reward"].append(self.ep_reward)
            self.metrics["length"].append(self.ep_length)
            for k, v in self.ep_metrics.items():
                if k in MAX_METRICS:
                    self.metrics[k].append(np.max(v))
                elif k in LAST_METRICS:  # Append the last value
                    self.metrics[k].append(v[-1])
                elif k in MEAN_METRICS:
                    self.metrics[k].append(np.mean(v))
                else:
                    self.metrics[k].append(np.sum(v))

            self.ep_length = 0
            self.ep_reward = 0
            self.ep_metrics = collections.defaultdict(list)

    def step(self, reward: float, info: Dict) -> None:
        self.ep_length += 1
        self.ep_reward += reward
        for k, v in info.items():
            if (isinstance(v, float) or np.isscalar(v)) and k not in EXCLUDE_METRICS:
                self.ep_metrics[k].append(v)

    def add(self, k: str, v: Any):
        self.metrics[k].append(v)

    def export(self) -> Dict:
        if self.ep_length > 0:
            # We have one remaining episode to log, make sure to get it.
            self.reset()
        metrics = {k: np.mean(v) for k, v in self.metrics.items()}
        metrics["reward_std"] = np.std(self.metrics["reward"])
        return metrics


def eval_multiple(env, model, path: str, step: int, eval_fns: List[str], eval_kwargs: List[Dict]):
    all_metrics = dict()
    for eval_fn, eval_kwarg in zip(eval_fns, eval_kwargs):
        metrics = locals()[eval_fn](env, model, path, step, **eval_kwarg)
        all_metrics.update(metrics)
    return all_metrics


def eval_policy(
    env: gym.Env,
    model,
    path: str,
    step: int,
    num_ep: int = 10,
    num_gifs: int = 0,
    width=200,
    height=200,
    every_n_frames: int = 2,
    terminate_on_success=False,
    history_length: int = 0,
    predict_kwargs: Optional[Dict] = None,
) -> Dict:
    metric_tracker = EvalMetricTracker()
    predict_kwargs = {} if predict_kwargs is None else predict_kwargs
    assert num_gifs <= num_ep, "Cannot save more gifs than eval ep."

    for i in range(num_ep):
        # Reset Metrics
        done = False
        ep_length, ep_reward = 0, 0
        frames = []
        save_gif = i < num_gifs
        render_kwargs = dict(mode="rgb_array", width=width, height=height) if save_gif else dict()
        obs = env.reset()
        if history_length > 0:
            obs = utils.unsqueeze(obs, 0)
        if save_gif:
            frames.append(env.render(**render_kwargs))
        metric_tracker.reset()
        while not done:
            batch = dict(obs=obs)
            if hasattr(env, "_max_episode_steps"):
                batch["horizon"] = env._max_episode_steps - ep_length
            with torch.no_grad():
                action = model.predict(batch, **predict_kwargs)
            if history_length > 0:
                action = action[-1]
            next_obs, reward, done, info = env.step(action)
            ep_reward += reward
            metric_tracker.step(reward, info)
            ep_length += 1
            if save_gif and ep_length % every_n_frames == 0:
                frames.append(env.render(**render_kwargs))
            if terminate_on_success and (info.get("success", False) or info.get("is_success", False)):
                done = True
            # Update the observation if we have history
            if history_length > 0:
                obs = utils.concatenate(obs, utils.unsqueeze(next_obs, 0), dim=0)
                if ep_length + 1 > history_length:
                    # Drop the last observation from the sequence
                    obs = utils.get_from_batch(obs, start=1, end=history_length + 1)
            else:
                obs = next_obs

        if hasattr(env, "get_normalized_score"):
            metric_tracker.add("score", env.get_normalized_score(ep_reward))

        if save_gif:
            gif_name = "vis-{}_ep-{}.gif".format(step, i)
            imageio.mimsave(os.path.join(path, gif_name), frames)

    return metric_tracker.export()
