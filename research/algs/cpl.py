import itertools
from typing import Any, Dict

import torch

from research.utils import utils

from .off_policy_algorithm import OffPolicyAlgorithm


def biased_bce_with_logits(adv1, adv2, y, bias=1.0):
    # Apply the log-sum-exp trick.
    # y = 1 if we prefer x2 to x1
    # We need to implement the numerical stability trick.

    logit21 = adv2 - bias * adv1
    logit12 = adv1 - bias * adv2
    max21 = torch.clamp(-logit21, min=0, max=None)
    max12 = torch.clamp(-logit12, min=0, max=None)
    nlp21 = torch.log(torch.exp(-max21) + torch.exp(-logit21 - max21)) + max21
    nlp12 = torch.log(torch.exp(-max12) + torch.exp(-logit12 - max12)) + max12
    loss = y * nlp21 + (1 - y) * nlp12
    loss = loss.mean()

    # Now compute the accuracy
    with torch.no_grad():
        accuracy = ((adv2 > adv1) == torch.round(y)).float().mean()

    return loss, accuracy


def biased_bce_with_scores(adv, scores, bias=1.0):
    # For now label clip does nothing.
    # Could try doing this asymetric with two sides, but found that it doesn't work well.

    idx = torch.argsort(scores, dim=0)
    adv_sorted = adv[idx]

    # Compute normalized loss
    logits = adv_sorted.unsqueeze(0) - bias * adv_sorted.unsqueeze(1)
    max_val = torch.clamp(-logits, min=0, max=None)
    loss = torch.log(torch.exp(-max_val) + torch.exp(-logits - max_val)) + max_val

    loss = torch.triu(loss, diagonal=1)
    mask = loss != 0.0
    loss = loss.sum() / mask.sum()

    with torch.no_grad():
        unbiased_logits = adv_sorted.unsqueeze(0) - adv_sorted.unsqueeze(1)
        accuracy = ((unbiased_logits > 0) * mask).sum() / mask.sum()

    return loss, accuracy


class CPL(OffPolicyAlgorithm):
    def __init__(
        self,
        *args,
        alpha: float = 1.0,
        contrastive_bias: float = 1.0,
        bc_coeff: float = 0.0,
        bc_data: str = "all",
        bc_steps: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        # Perform checks on values
        assert "encoder" in self.network.CONTAINERS
        assert "actor" in self.network.CONTAINERS
        assert contrastive_bias > 0.0 and contrastive_bias <= 1.0
        self.alpha = alpha
        self.contrastive_bias = contrastive_bias
        self.bc_data = bc_data
        self.bc_steps = bc_steps
        self.bc_coeff = bc_coeff

    def setup_optimizers(self) -> None:
        params = itertools.chain(self.network.actor.parameters(), self.network.encoder.parameters())
        groups = utils.create_optim_groups(params, self.optim_kwargs)
        self.optim["actor"] = self.optim_class(groups)

    def setup_schedulers(self, do_nothing=True):
        if do_nothing:
            # Set schedulers that just return 1.0 -- ignore during BC steps.
            for k in self.schedulers_class.keys():
                if k in self.optim:
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
        obs = self.network.encoder(obs)
        dist = self.network.actor(obs)
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

        # Step 2: Compute the advantages.
        adv = self.alpha * lp
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
