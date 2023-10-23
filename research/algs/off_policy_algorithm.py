import datetime
import functools
import os
import sys
import tempfile
from abc import abstractmethod
from typing import Any, Dict, Optional, Union

import gym
import numpy as np
import torch

from research.datasets import ReplayBuffer
from research.datasets.replay_buffer import storage
from research.envs.base import EmptyEnv
from research.networks.base import ModuleContainer
from research.utils import runners, utils

from .base import Algorithm


class OffPolicyAlgorithm(Algorithm):
    def __init__(
        self,
        *args,
        offline_steps: int = 0,  # Run fully offline by setting to -1
        random_steps: int = 1000,
        async_runner_ep_lag: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.offline_steps = offline_steps
        self.random_steps = random_steps
        self.async_runner_ep_lag = async_runner_ep_lag

    def setup_datasets(self, env: gym.Env, total_steps: int):
        super().setup_datasets(env, total_steps)
        # Assign the correct update function based on what is passed in.
        if env is None or isinstance(env, EmptyEnv) or self.offline_steps < 0:
            self.env_step = self._empty_step
        elif isinstance(env, runners.AsyncEnv):
            self._episode_reward = 0
            self._episode_length = 0
            self._num_ep = 0
            self._env_steps = 0
            self._resetting = True
            env.reset_send()  # Ask the env to start resetting.
            self.env_step = self._async_env_step
        elif isinstance(env, runners.MPRunner):
            assert isinstance(self.dataset, ReplayBuffer), "must use replaybuffer for MP RUnner."
            assert self.dataset.distributed, "ReplayBuffer must be distributed for use with Fully MPRunner."
            # Launch the runner subprocess.
            self._eps_since_last_checkpoint = 0
            self._checkpoint_dir = tempfile.mkdtemp(prefix="checkpoints_")
            assert self.offline_steps <= 0, "MPRunner does not currently support offline to online."
            env.start(
                fn=_off_policy_collector_subprocess,
                checkpoint_path=self._checkpoint_dir,
                storage_path=self.dataset.storage_path,
                random_steps=self.random_steps,
                exclude_keys=self.dataset.exclude_keys,
                total_steps=total_steps,
            )
            self.env_step = self._runner_env_step
        elif isinstance(env, gym.Env):
            # Setup Env Metrics
            self._current_obs = env.reset()
            self._episode_reward = 0
            self._episode_length = 0
            self._num_ep = 0
            self._env_steps = 0
            # Note that currently the very first (s, a) pair is thrown away because
            # we don't add to the dataset here.
            # This was done for better compatibility for offline to online learning.
            self.dataset.add(obs=self._current_obs)  # add the first observation.
            self.env_step = self._env_step
        else:
            raise ValueError("Invalid env passed")

    def _empty_step(self, env: gym.Env, step: int, total_steps: int) -> Dict:
        return dict()

    def _env_step(self, env: gym.Env, step: int, total_steps: int) -> Dict:
        # Return if env is Empty or we we aren't at every env_freq steps
        if step <= self.offline_steps:
            # Purposefully set to nan so we write CSV log.
            return dict(steps=self._env_steps, reward=-np.inf, length=np.inf, num_ep=self._num_ep)

        if step < self.random_steps:
            action = env.action_space.sample()
        else:
            self.eval()
            action = self._get_train_action(self._current_obs, step, total_steps)
            self.train()
        if isinstance(env.action_space, gym.spaces.Box):
            action = np.clip(action, env.action_space.low, env.action_space.high)

        next_obs, reward, done, info = env.step(action)
        self._env_steps += 1
        self._episode_length += 1
        self._episode_reward += reward

        if "discount" in info:
            discount = info["discount"]
        elif hasattr(env, "_max_episode_steps") and self._episode_length == env._max_episode_steps:
            discount = 1.0
        else:
            discount = 1 - float(done)

        # Store the consequences.
        self.dataset.add(obs=next_obs, action=action, reward=reward, done=done, discount=discount)

        if done:
            self._num_ep += 1
            # Compute metrics
            metrics = dict(
                steps=self._env_steps, reward=self._episode_reward, length=self._episode_length, num_ep=self._num_ep
            )
            # Reset the environment
            self._current_obs = env.reset()
            self.dataset.add(obs=self._current_obs)  # Add the first timestep
            self._episode_length = 0
            self._episode_reward = 0
            return metrics
        else:
            self._current_obs = next_obs
            return dict(steps=self._env_steps)

    def _async_env_step(self, env: gym.Env, step: int, total_steps: int) -> Dict:
        # Recieve Data from the last step and add to buffer. Should only call recv!
        if self._resetting:
            self._current_obs = env.reset_recv()
            self._num_ep += 1
            self._episode_length = 0
            self._episode_reward = 0
            self.dataset.add(obs=self._current_obs)
            self._resetting = False
            done = False
        else:
            self._current_obs, reward, done, info = env.step_recv()
            self._env_steps += 1
            self._episode_length += 1
            self._episode_reward += reward
            self.dataset.add(
                obs=self._current_obs, action=self._current_action, reward=reward, done=done, discount=info["discount"]
            )

        # Send data for the next step and return metrics. Should only call send!
        if done:
            # If the episode terminated, then we need to reset and send the reset message
            self._resetting = True
            env.reset_send()
            return dict(
                steps=self._env_steps, reward=self._episode_reward, length=self._episode_length, num_ep=self._num_ep
            )
        else:
            # Otherwise, compute the action we should take and send it.
            self._resetting = False
            if step < self.random_steps:
                self._current_action = env.action_space.sample()
            else:
                self.eval()
                self._current_action = self._get_train_action(self._current_obs, step, total_steps)
                self.train()
            if isinstance(env.action_space, gym.spaces.Box):
                self._current_action = np.clip(self._current_action, env.action_space.low, env.action_space.high)
            env.step_send(self._current_action)
            return dict(steps=self._env_steps)

    def _runner_env_step(self, env: gym.Env, step: int, total_steps: int) -> Dict:
        # All we do is check the pipe to see if there is data!
        metrics = env()
        if len(metrics) > 0:
            # If the metrics are non-empty, then it means that we have completed an episode.
            # As such, decrement the counter
            self._eps_since_last_checkpoint += 1
        if self._eps_since_last_checkpoint == self.async_runner_ep_lag:
            self.save(self._checkpoint_dir, str(step), dict(step=step))
            self._eps_since_last_checkpoint = 0
        return metrics

    @abstractmethod
    def _get_train_action(self, obs: Any, step: int, total_steps: int) -> np.ndarray:
        raise NotImplementedError

    @functools.cached_property
    def action_range(self):
        action_range = (self.processor.action_space.low, self.processor.action_space.high)
        return utils.to_device(utils.to_tensor(action_range), self.device)

    def _predict(
        self, batch: Dict, sample: bool = False, noise: float = 0.0, noise_clip: Optional[float] = None, temperature=1.0
    ) -> torch.Tensor:
        with torch.no_grad():
            if isinstance(self.network, ModuleContainer) and "encoder" in self.network.CONTAINERS:
                obs = self.network.encoder(batch["obs"])
            else:
                obs = batch["obs"]

            # Could be: Logits (discrete), Float (continuous), or torch Dist
            dist = self.network.actor(obs)

            if isinstance(self.processor.action_space, gym.spaces.Box):
                if isinstance(dist, torch.distributions.Independent):
                    # Guassian Distribution
                    action = dist.sample() if sample else dist.base_dist.loc

                elif isinstance(dist, torch.distributions.MixtureSameFamily):
                    # Mixture of Gaussians.
                    if sample:
                        action = dist.sample()
                    else:
                        # Robomimic always samples from the Categorical, but then does the mixture deterministically.
                        loc = dist.component_distribution.base_dist.loc
                        category = dist.mixture_distribution.sample()

                        # Expand to add Mixture Dim, Action Dim
                        es = dist.component_distribution.event_shape
                        mix_sample_r = category.reshape(category.shape + torch.Size([1] * (len(es) + 1)))
                        mix_sample_r = mix_sample_r.repeat(torch.Size([1] * len(category.shape)) + torch.Size([1]) + es)
                        action = torch.gather(loc, len(dist.batch_shape), mix_sample_r)
                        action = action.squeeze(len(dist.batch_shape))

                elif torch.is_tensor(dist):
                    action = dist

                else:
                    raise ValueError("Model output incompatible with default _predict.")

                if noise > 0.0:
                    eps = noise * torch.randn_like(action)
                    if noise_clip is not None:
                        eps = torch.clamp(eps, -noise_clip, noise_clip)
                    action = action + eps
                action = action.clamp(*self.action_range)
                return action

            elif isinstance(self.processor.action_space, gym.spaces.Discrete):
                logits = dist.logits if isinstance(dist, torch.distributions.Categorical) else dist
                if sample:
                    action = torch.distributions.Categorical(logits=logits / temperature).sample()
                else:
                    action = logits.argmax(dim=-1)

                return action

            else:
                raise ValueError("Complex action_space incompatible with default _predict.")


def _off_policy_collector_subprocess(
    env_fn,
    queue,
    config_path: str,
    checkpoint_path: str,
    storage_path: str,
    exclude_keys: Optional[Optional[list]] = None,
    device: Union[str, torch.device] = "auto",
    random_steps: int = 0,
    total_steps: int = 0,
):
    """
    This subprocess loads a train environemnt.
    It then collects episodes with a loaded policy and saves them to disk.
    Afterwards, we check to see if there is an updated policy that we can use.
    """
    try:
        env = env_fn()
        # Load the model
        from research.utils.config import Config

        config = Config.load(config_path)
        config = config.parse()
        model = config.get_model(observation_space=env.observation_space, action_space=env.action_space, device=device)
        model.eval()

        # Compute the buffer space
        buffer_space = {
            "obs": env.observation_space,
            "action": env.action_space,
            "reward": 0.0,
            "done": False,
            "discount": 1.0,
        }
        exclude_keys = [] if exclude_keys is None else exclude_keys
        flattened_buffer_space = utils.flatten_dict(buffer_space)
        for k in exclude_keys:
            del flattened_buffer_space[k]

        def make_dummy_transition(obs):
            return {
                "obs": obs,
                "action": env.action_space.sample(),
                "reward": 0.0,
                "discount": 1.0,
                "done": False,
            }

        # Metrics:
        num_ep = 0
        env_steps = 0
        current_checkpoint = None

        # Get the evaluation function.
        while True:
            # First, look for a checkpoint.
            checkpoints = os.listdir(checkpoint_path)
            if len(checkpoints) > 0:
                # Sort the the checkpoints by path
                checkpoints = sorted(checkpoints, key=lambda x: int(x[:-3]))
                checkpoints = [os.path.join(checkpoint_path, checkpoint) for checkpoint in checkpoints]
                if checkpoints[-1] != current_checkpoint and os.path.getsize(checkpoints[-1]) > 0:
                    try:
                        _ = model.load(checkpoints[-1])  # load the most recent one
                        # Remove all checkpoints that are not equal to the current one.
                        current_checkpoint = checkpoints[-1]
                        for checkpoint in checkpoints[:-1]:  # Ignore the last checkpoint, we loaded it.
                            os.remove(checkpoint)
                    except (EOFError, RuntimeError):
                        _ = model.load(current_checkpoint)

            # Then, collect an episode
            current_ep = {k: list() for k in flattened_buffer_space.keys()}
            current_ep = utils.nest_dict(current_ep)
            obs = env.reset()
            utils.append(current_ep, make_dummy_transition(obs))
            done = False

            while not done:
                if env_steps < random_steps:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = model._get_train_action(obs, env_steps, total_steps)

                obs, reward, done, info = env.step(action)
                env_steps += 1

                if "discount" in info:
                    discount = info["discount"]
                elif hasattr(env, "_max_episode_steps") and len(current_ep["done"]) - 1 == env._max_episode_steps:
                    discount = 1.0
                else:
                    discount = 1 - float(done)
                transition = dict(obs=obs, action=action, reward=reward, done=done, discount=discount)
                utils.append(current_ep, transition)

            # The episode has terminated.
            num_ep += 1
            metrics = dict(
                steps=env_steps, reward=np.sum(current_ep["reward"]), length=len(current_ep["done"]) - 1, num_ep=num_ep
            )
            queue.put(metrics)
            # Timestamp it and add the ep idx (num ep - 1 so we start at zero.)
            ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
            ep_len = len(current_ep["done"])
            ep_filename = f"{ts}_{num_ep - 1}_{ep_len}.npz"
            storage.save_data(current_ep, os.path.join(storage_path, ep_filename))

    except KeyboardInterrupt:
        print("[research] OffPolicy Collector sent interrupt.")
        queue.put(None)  # Add None in the queue, ie failure!
    except Exception as e:
        print("[research] OffPolicy Collector Subprocess encountered exception.")
        print(e)
        print(sys.exec_info()[:2])
        queue.put(None)  # Add None in the queue, ie failure!
    finally:
        env.close()  # Close the env to prevent hanging threads.
