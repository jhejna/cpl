import itertools
from typing import Any, Dict, Type

import torch

from research.networks.base import ActorPolicy
from research.utils import utils

from .cpl import biased_bce_with_logits, biased_bce_with_scores
from .off_policy_algorithm import OffPolicyAlgorithm


class CPL_KL(OffPolicyAlgorithm):
    def __init__(
        self,
        *args,
        alpha: float = 1.0,
        contrastive_bias: float = 1.0,
        bc_coeff: float = 0.0,
        bc_data: str = "all",
        bc_steps: int = 10000,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        # Perform checks on values
        assert "encoder" in self.network.CONTAINERS
        assert "actor" in self.network.CONTAINERS
        assert bc_steps > 0
        assert contrastive_bias > 0.0 and contrastive_bias <= 1.0
        self.alpha = alpha
        self.contrastive_bias = contrastive_bias
        self.bc_data = bc_data
        self.bc_coeff = bc_coeff
        self.bc_steps = bc_steps

    def setup_network(self, network_class: Type[torch.nn.Module], network_kwargs: Dict) -> None:
        self.network = network_class(
            self.processor.observation_space, self.processor.action_space, **network_kwargs
        ).to(self.device)
        self.reference_network = ActorPolicy(
            self.processor.observation_space, self.processor.action_space, **network_kwargs
        ).to(self.device)
        for param in self.reference_network.parameters():
            param.requires_grad = False

    def setup_optimizers(self) -> None:
        params = itertools.chain(self.network.actor.parameters(), self.network.encoder.parameters())
        groups = utils.create_optim_groups(params, self.optim_kwargs)
        self.optim["actor"] = self.optim_class(groups)

    def setup_schedulers(self, do_nothing=True):
        if do_nothing:
            # Set schedulers that just return 1.0 -- ignore during BC steps.
            for k in self.schedulers_class.keys():
                self.schedulers[k] = torch.optim.lr_scheduler.LambdaLR(self.optim[k], lr_lambda=lambda x: 1.0)
        else:
            self.schedulers = {}
            super().setup_schedulers()

    def _get_cpl_loss(self, batch):
        if isinstance(batch, dict) and "label" in batch:
            obs = torch.cat((batch["obs_1"], batch["obs_2"]), dim=0)
            action = torch.cat((batch["action_1"], batch["action_2"]), dim=0)
        else:
            assert "score" in batch
            obs, action = batch["obs"], batch["action"]

        # Step 1: Compute the log probabilities
        dist = self.network.actor(self.network.encoder(obs))
        if isinstance(dist, torch.distributions.Distribution):
            lp = dist.log_prob(action)
        else:
            assert dist.shape == action.shape
            # For independent gaussian with unit var, logprob reduces to MSE.
            lp = -torch.square(dist - action).sum(dim=-1)

        # Compute the BC Loss from the log probabilities.
        # In some cases we might want to only do this on the positive data.
        if self.bc_data == "pos":
            lp1, lp2 = torch.chunk(lp, 2, dim=0)  # Should return two (B, S)
            lp_pos = torch.cat((lp1[batch["label"] <= 0.5], lp2[batch["label"] >= 0.5]), dim=0)
            bc_loss = (-lp_pos).mean()  # We have a full mask when using feedback data.
        else:
            bc_loss = (-lp).mean()

        # Compute the reference log probabilities
        with torch.no_grad():
            self.reference_network.eval()
            ref_dist = self.reference_network.actor(self.reference_network.encoder(obs))
            if isinstance(dist, torch.distributions.Distribution):
                ref_lp = ref_dist.log_prob(action)
            else:
                assert dist.shape == action.shape
                ref_lp = -torch.square(ref_dist - action).sum(dim=-1)

        # Step 2: Compute the advantages.
        adv = self.alpha * (lp - ref_lp)
        segment_adv = adv.sum(dim=-1)

        # Step 3: Compute the Loss.
        if "score" in batch:
            cpl_loss, accuracy = biased_bce_with_scores(segment_adv, batch["score"].float(), bias=self.contrastive_bias)
        else:
            # Otherwise we update directly on the preference data with the standard CE loss
            adv1, adv2 = torch.chunk(segment_adv, 2, dim=0)
            cpl_loss, accuracy = biased_bce_with_logits(adv1, adv2, batch["label"].float(), bias=self.contrastive_bias)
        return cpl_loss, bc_loss, accuracy

    def train_step(self, batch: Dict, step: int, total_steps: int) -> Dict:
        cpl_loss, bc_loss, accuracy = self._get_cpl_loss(batch)

        # Train with only BC loss for bc_steps
        if step < self.bc_steps:
            loss = bc_loss
            cpl_loss, accuracy = torch.tensor(0.0), torch.tensor(0.0)
        else:
            loss = cpl_loss + self.bc_coeff * bc_loss

        self.optim["actor"].zero_grad()
        loss.backward()
        self.optim["actor"].step()

        if step == self.bc_steps - 1:  # Switch to optimizing CPL loss here.
            del self.optim["actor"]

            self.reference_network.encoder.load_state_dict(self.network.encoder.state_dict())
            self.reference_network.actor.load_state_dict(self.network.actor.state_dict())

            # Reset the optim and LR schedule
            params = itertools.chain(self.network.actor.parameters(), self.network.encoder.parameters())
            groups = utils.create_optim_groups(params, self.optim_kwargs)
            self.optim["actor"] = self.optim_class(groups)
            self.setup_schedulers(do_nothing=False)  # actually start the schedulers.

        return dict(cpl_loss=cpl_loss.item(), bc_loss=bc_loss.item(), accuracy=accuracy.item())

    def validation_step(self, batch: Any) -> Dict:
        with torch.no_grad():
            cpl_loss, bc_loss, accuracy = self._get_cpl_loss(batch)
        return dict(cpl_loss=cpl_loss.item(), bc_loss=bc_loss.item(), accuracy=accuracy.item())

    def _get_train_action(self, obs: Any, step: int, total_steps: int):
        batch = dict(obs=obs)
        with torch.no_grad():
            action = self.predict(batch, is_batched=False, sample=True)
        return action
