import itertools
from typing import Any, Dict, Optional

import gym
import torch

from research.utils import utils

from .off_policy_algorithm import OffPolicyAlgorithm

IGNORE_INDEX = -100


class BehaviorCloning(OffPolicyAlgorithm):
    """
    BC Implementation.
    Uses MSE loss for continuous, and CE for discrete
    """

    def __init__(
        self, *args, grad_norm_clip: Optional[float] = None, bc_data: str = "all", bc_all_steps: int = 0, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        assert "encoder" in self.network.CONTAINERS
        assert "actor" in self.network.CONTAINERS
        assert isinstance(self.action_space, gym.spaces.Box)
        assert bc_data in {"all", "pos"}
        self.bc_data = bc_data
        self.bc_all_steps = bc_all_steps
        self.grad_norm_clip = grad_norm_clip

    def setup_optimizers(self) -> None:
        params = itertools.chain(self.network.actor.parameters(), self.network.encoder.parameters())
        groups = utils.create_optim_groups(params, self.optim_kwargs)
        self.optim["actor"] = self.optim_class(groups)

    def _get_bc_loss(self, obs, action):
        z = self.network.encoder(obs)
        dist = self.network.actor(z)
        if isinstance(dist, torch.distributions.Distribution):
            loss = -dist.log_prob(action)  # NLL Loss
        elif torch.is_tensor(dist) and isinstance(self.processor.action_space, gym.spaces.Box):
            loss = torch.nn.functional.mse_loss(dist, action, reduction="none").sum(dim=-1)  # MSE Loss
        elif torch.is_tensor(dist) and isinstance(self.processor.action_space, gym.spaces.Discrete):
            loss = torch.nn.functional.cross_entropy(dist, action, ignore_index=IGNORE_INDEX, reduction="none")
        else:
            raise ValueError("Invalid Policy output")

        return loss.mean()  # Simple average.

    def train_step(self, batch: Dict, step: int, total_steps: int) -> Dict:
        if isinstance(batch, dict) and "label" in batch:
            if self.bc_data == "pos" and step >= self.bc_all_steps:
                prefer_1 = batch["label"] <= 0.5
                prefer_2 = batch["label"] >= 0.5
                obs = torch.cat((batch["obs_1"][prefer_1], batch["obs_2"][prefer_2]), dim=0)
                action = torch.cat((batch["action_1"][prefer_1], batch["action_2"][prefer_2]), dim=0)
            else:
                obs = torch.cat((batch["obs_1"], batch["obs_2"]), dim=0)
                action = torch.cat((batch["action_1"], batch["action_2"]), dim=0)
        else:
            assert "obs" in batch
            assert "action" in batch
            assert self.bc_data != "pos", "Cannot select only pos data for replay dataset."
            obs, action = batch["obs"], batch["action"]

        loss = self._get_bc_loss(obs, action)
        self.optim["actor"].zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_norm_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_norm_clip)
        self.optim["actor"].step()

        return dict(loss=loss.item())

    def validation_step(self, batch: Any) -> Dict:
        if isinstance(batch, dict) and "label" in batch:
            if self.bc_data == "pos":
                prefer_1 = batch["label"] <= 0.5
                prefer_2 = batch["label"] >= 0.5
                obs = torch.cat((batch["obs_1"][prefer_1], batch["obs_2"][prefer_2]), dim=0)
                action = torch.cat((batch["action_1"][prefer_1], batch["action_2"][prefer_2]), dim=0)
            else:
                obs = torch.cat((batch["obs_1"], batch["obs_2"]), dim=0)
                action = torch.cat((batch["action_1"], batch["action_2"]), dim=0)
        else:
            assert "obs" in batch
            assert "action" in batch
            assert self.bc_data != "pos", "Cannot select only pos data for replay dataset."
            obs, action = batch["obs"], batch["action"]

        with torch.no_grad():
            loss = self._get_bc_loss(obs, action)

        return dict(loss=loss.item())

    def _get_train_action(self, obs: Any, step: int, total_steps: int):
        batch = dict(obs=obs)
        with torch.no_grad():
            action = self.predict(batch, is_batched=False, sample=True)
        return action
