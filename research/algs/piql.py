import itertools
from typing import Any, Dict, Optional, Type

import torch

from research.networks.base import ActorCriticValueRewardPolicy

from .off_policy_algorithm import OffPolicyAlgorithm


def iql_loss(pred, target, expectile=0.5):
    err = target - pred
    weight = torch.abs(expectile - (err < 0).float())
    return weight * torch.square(err)


class PIQL(OffPolicyAlgorithm):
    def __init__(
        self,
        *args,
        tau: float = 0.005,
        target_freq: int = 1,
        expectile: Optional[float] = None,
        beta: float = 1,
        clip_score: float = 100.0,
        reward_steps: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert isinstance(self.network, ActorCriticValueRewardPolicy)
        self.tau = tau
        self.target_freq = target_freq
        self.expectile = expectile
        self.beta = beta
        self.clip_score = clip_score
        self.reward_steps = reward_steps

        self.reward_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    def setup_network(self, network_class: Type[torch.nn.Module], network_kwargs: Dict) -> None:
        self.network = network_class(
            self.processor.observation_space, self.processor.action_space, **network_kwargs
        ).to(self.device)
        self.target_network = network_class(
            self.processor.observation_space, self.processor.action_space, **network_kwargs
        ).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        for param in self.target_network.parameters():
            param.requires_grad = False

    def setup_optimizers(self) -> None:
        # Default optimizer initialization
        network_keys = ("actor", "critic", "value", "reward")
        default_kwargs = {k: v for k, v in self.optim_kwargs.items() if k not in network_keys}
        assert all([isinstance(self.optim_kwargs.get(k, dict()), dict) for k in network_keys])

        # Update the encoder with the actor. This does better for weighted imitation policy objectives.
        actor_kwargs = default_kwargs.copy()
        actor_kwargs.update(self.optim_kwargs.get("actor", dict()))
        actor_params = itertools.chain(self.network.actor.parameters(), self.network.encoder.parameters())
        self.optim["actor"] = self.optim_class(actor_params, **actor_kwargs)

        critic_kwargs = default_kwargs.copy()
        critic_kwargs.update(self.optim_kwargs.get("critic", dict()))
        self.optim["critic"] = self.optim_class(self.network.critic.parameters(), **critic_kwargs)

        value_kwargs = default_kwargs.copy()
        value_kwargs.update(self.optim_kwargs.get("value", dict()))
        self.optim["value"] = self.optim_class(self.network.value.parameters(), **value_kwargs)

        reward_kwargs = default_kwargs.copy()
        reward_kwargs.update(self.optim_kwargs.get("reward", dict()))
        self.optim["reward"] = self.optim_class(self.network.reward.parameters(), **reward_kwargs)

    def train_step(self, batch: Dict, step: int, total_steps: int) -> Dict:
        assert isinstance(batch, dict) and "label" in batch, "Feedback batch must be used for efficient pref_iql"
        obs = torch.cat([batch["obs_1"], batch["obs_2"]], dim=0)  # (B, S+1)
        action = torch.cat([batch["action_1"], batch["action_2"]], dim=0)  # (B, S+1)
        discount = torch.cat((batch["discount_1"], batch["discount_2"]), dim=0)  # (B, S+1)

        if step < self.reward_steps:
            self.network.reward.train()
            reward = self.network.reward(obs, action)

            r1, r2 = torch.chunk(reward.sum(dim=-1), 2, dim=1)  # Should return two (E, B)
            logits = r2 - r1
            labels = batch["label"].float().unsqueeze(0).expand_as(logits)

            assert labels.shape == logits.shape
            reward_loss = self.reward_criterion(logits, labels).mean()

            with torch.no_grad():
                reward_accuracy = ((r2 > r1) == torch.round(labels)).float().mean()

            self.optim["reward"].zero_grad(set_to_none=True)
            reward_loss.backward()
            self.optim["reward"].step()

            reward = reward.detach().mean(dim=0)
        else:
            with torch.no_grad():
                reward = self.network.reward(obs, action).mean(dim=0)

        # Encode everything
        obs = self.network.encoder(obs)
        next_obs = obs[:, 1:].detach()
        obs = obs[:, :-1]
        action = action[:, :-1]
        discount = discount[:, :-1]
        reward = reward[:, :-1]

        with torch.no_grad():
            target_q = self.target_network.critic(obs, action)
            target_q = torch.min(target_q, dim=0)[0]
        vs = self.network.value(obs.detach())
        v_loss = iql_loss(vs, target_q.unsqueeze(0).expand_as(vs), self.expectile).mean()

        self.optim["value"].zero_grad(set_to_none=True)
        v_loss.backward()
        self.optim["value"].step()

        # Next, update the actor. We detach and use the old value, v for computational efficiency
        # and use the target_q value though the JAX IQL recomputes both
        # Pytorch IQL versions have not.
        with torch.no_grad():
            adv = target_q - torch.mean(vs, dim=0)  # min trick is not used on value.
            exp_adv = torch.exp(adv / self.beta)
            if self.clip_score is not None:
                exp_adv = torch.clamp(exp_adv, max=self.clip_score)

        dist = self.network.actor(obs)  # Use encoder gradients for the actor.
        if isinstance(dist, torch.distributions.Distribution):
            bc_loss = -dist.log_prob(action)
        elif torch.is_tensor(dist):
            assert dist.shape == action.shape
            bc_loss = torch.nn.functional.mse_loss(dist, action, reduction="none").sum(dim=-1)
        else:
            raise ValueError("Invalid policy output provided")
        actor_loss = (exp_adv * bc_loss).mean()

        self.optim["actor"].zero_grad(set_to_none=True)
        actor_loss.backward()
        self.optim["actor"].step()

        # Next, Finally update the critic
        with torch.no_grad():
            next_vs = self.network.value(next_obs)
            next_v = torch.mean(next_vs, dim=0, keepdim=True)  # Min trick is not used on value.
            target = reward + discount * next_v  # use the predicted reward.
        qs = self.network.critic(obs.detach(), action)
        q_loss = torch.nn.functional.mse_loss(qs, target.expand_as(qs), reduction="none").mean()

        self.optim["critic"].zero_grad(set_to_none=True)
        q_loss.backward()
        self.optim["critic"].step()

        metrics = dict(
            q_loss=q_loss.item(),
            v_loss=v_loss.item(),
            actor_loss=actor_loss.item(),
            v=vs.mean().item(),
            q=qs.mean().item(),
            adv=adv.mean().item(),
            reward=reward.mean().item(),
        )

        if step < self.reward_steps:
            metrics["reward_loss"] = reward_loss.item()
            metrics["reward_accuracy"] = reward_accuracy.item()

        # Update the networks. These are done in a stack to support different grad options for the encoder.
        if step % self.target_freq == 0:
            with torch.no_grad():
                # Only run on the critic and encoder, those are the only weights we update.
                for param, target_param in zip(
                    self.network.critic.parameters(), self.target_network.critic.parameters()
                ):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return metrics

    def validation_step(self, batch: Dict) -> Dict:
        # Compute the loss
        if isinstance(batch, (tuple, list)) and "label" in batch[1]:
            feedback_batch = batch[1]
            with torch.no_grad():
                reward_loss, reward_accuracy, reward_pred = self._get_reward_loss(feedback_batch)
            return dict(
                reward_loss=reward_loss.item(), reward_accuracy=reward_accuracy.item(), reward=reward_pred.mean().item()
            )
        else:
            return dict()

    def _get_train_action(self, obs: Any, step: int, total_steps: int):
        batch = dict(obs=obs)
        with torch.no_grad():
            action = self.predict(batch, is_batched=False, sample=True)
        return action
