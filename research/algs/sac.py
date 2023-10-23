import itertools
from typing import Any, Dict, Type

import numpy as np
import torch

from research.networks.base import ActorCriticPolicy

from .off_policy_algorithm import OffPolicyAlgorithm


class SAC(OffPolicyAlgorithm):
    def __init__(
        self,
        *args,
        tau: float = 0.005,
        init_temperature: float = 0.1,
        critic_freq: int = 1,
        actor_freq: int = 1,
        target_freq: int = 2,
        freeze_actor_steps: int = 0,
        bc_coeff: float = 0.0,
        bc_steps: int = 0,
        env_steps=None,  #  This was used earlier and is saved in some configs. Ignore it.
        **kwargs,
    ):
        # Save values needed for network setup.
        self.init_temperature = init_temperature
        super().__init__(*args, **kwargs)
        assert isinstance(self.network, ActorCriticPolicy)

        # Save extra parameters
        self.tau = tau
        self.critic_freq = critic_freq
        self.actor_freq = actor_freq
        self.target_freq = target_freq
        self.freeze_actor_steps = freeze_actor_steps
        self.bc_coeff = bc_coeff
        self.bc_steps = bc_steps
        self.target_entropy = -np.prod(self.processor.action_space.low.shape)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def setup_network(self, network_class: Type[torch.nn.Module], network_kwargs: Dict) -> None:
        # Setup network and target network
        self.network = network_class(
            self.processor.observation_space, self.processor.action_space, **network_kwargs
        ).to(self.device)
        self.target_network = network_class(
            self.processor.observation_space, self.processor.action_space, **network_kwargs
        ).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        for param in self.target_network.parameters():
            param.requires_grad = False

        # Setup the log alpha
        log_alpha = torch.tensor(np.log(self.init_temperature), dtype=torch.float).to(self.device)
        self.log_alpha = torch.nn.Parameter(log_alpha, requires_grad=True)

    def setup_optimizers(self) -> None:
        # Default optimizer initialization
        self.optim["actor"] = self.optim_class(self.network.actor.parameters(), **self.optim_kwargs)
        # Update the encoder with the critic.
        critic_params = itertools.chain(self.network.critic.parameters(), self.network.encoder.parameters())
        self.optim["critic"] = self.optim_class(critic_params, **self.optim_kwargs)
        self.optim["log_alpha"] = self.optim_class([self.log_alpha], **self.optim_kwargs)

    def _update_critic(self, batch: Dict) -> Dict:
        with torch.no_grad():
            dist = self.network.actor(batch["next_obs"])
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action)
            target_qs = self.target_network.critic(batch["next_obs"], next_action)
            target_v = torch.min(target_qs, dim=0)[0] - self.alpha.detach() * log_prob
            target_q = batch["reward"] + batch["discount"] * target_v

        qs = self.network.critic(batch["obs"], batch["action"])
        q_loss = torch.nn.functional.mse_loss(qs, target_q.expand(qs.shape[0], -1), reduction="none").mean()

        self.optim["critic"].zero_grad(set_to_none=True)
        q_loss.backward()
        self.optim["critic"].step()

        return dict(q_loss=q_loss.item(), target_q=target_q.mean().item())

    def _update_actor_and_alpha_bc(self, batch: Dict) -> Dict:
        obs = batch["obs"].detach()  # Detach the encoder so it isn't updated.
        dist = self.network.actor(obs)
        bc_loss = -dist.log_prob(torch.clamp(batch["action"], min=-0.995, max=0.995)).mean()  # Simple NLL loss.

        self.optim["actor"].zero_grad(set_to_none=True)
        bc_loss.backward()
        self.optim["actor"].step()

        return dict(
            actor_loss=bc_loss.item(),
            entropy=0.0,
            alpha_loss=0.0,
            alpha=self.alpha.detach().item(),
        )

    def _update_actor_and_alpha(self, batch: Dict) -> Dict:
        obs = batch["obs"].detach()  # Detach the encoder so it isn't updated.
        dist = self.network.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        qs = self.network.critic(obs, action)
        q = torch.min(qs, dim=0)[0]

        actor_loss = (self.alpha.detach() * log_prob - q).mean()
        if self.bc_coeff > 0.0:
            bc_loss = -dist.log_prob(torch.clamp(batch["action"], min=-0.995, max=0.995)).mean()  # Simple NLL loss.
            actor_loss = actor_loss + self.bc_coeff * bc_loss

        self.optim["actor"].zero_grad(set_to_none=True)
        actor_loss.backward()
        self.optim["actor"].step()
        entropy = -log_prob.mean()

        # Update the learned temperature
        self.optim["log_alpha"].zero_grad(set_to_none=True)
        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.optim["log_alpha"].step()

        return dict(
            actor_loss=actor_loss.item(),
            entropy=entropy.item(),
            alpha_loss=alpha_loss.item(),
            alpha=self.alpha.detach().item(),
        )

    def train_step(self, batch: Dict, step: int, total_steps: int) -> Dict:
        all_metrics = {}

        if "obs" not in batch or step < self.random_steps:
            return all_metrics

        batch["obs"] = self.network.encoder(batch["obs"])
        with torch.no_grad():
            batch["next_obs"] = self.target_network.encoder(batch["next_obs"])

        if step % self.critic_freq == 0:
            metrics = self._update_critic(batch)
            all_metrics.update(metrics)

        if step - self.random_steps < self.bc_steps and step > self.freeze_actor_steps:
            metrics = self._update_actor_and_alpha_bc(batch)
            all_metrics.update(metrics)
        elif step % self.actor_freq == 0 and step > self.freeze_actor_steps:
            metrics = self._update_actor_and_alpha(batch)
            all_metrics.update(metrics)

        if step % self.target_freq == 0:
            # Only update the critic and encoder for speed. Ignore the actor.
            with torch.no_grad():
                for param, target_param in zip(
                    self.network.encoder.parameters(), self.target_network.encoder.parameters()
                ):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(
                    self.network.critic.parameters(), self.target_network.critic.parameters()
                ):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return all_metrics

    def _get_train_action(self, obs: Any, step: int, total_steps: int) -> np.ndarray:
        batch = dict(obs=obs)
        with torch.no_grad():
            action = self.predict(batch, is_batched=False, sample=True)
        return action
