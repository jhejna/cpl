from collections.abc import Iterable
from functools import partial
from typing import Optional, Type

import gym
import torch
from torch import distributions as D
from torch import nn

from .common import MLP, EnsembleMLP, LinearEnsemble


def weight_init(m: nn.Module, gain: int = 1) -> None:
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain=gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    if isinstance(m, LinearEnsemble):
        for i in range(m.ensemble_size):
            # Orthogonal initialization doesn't care about which axis is first
            # Thus, we can just use ortho init as normal on each matrix.
            nn.init.orthogonal_(m.weight.data[i], gain=gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class MLPEncoder(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        hidden_layers: Iterable[int] = (256, 256),
        ortho_init: bool = False,
        **kwargs,
    ):
        assert isinstance(observation_space, gym.spaces.Box) and len(observation_space.shape) == 1
        assert len(hidden_layers) > 1, "Must have at least one hidden layer for a shared MLP Extractor"
        self.mlp = MLP(observation_space.shape[0], hidden_layers[-1], hidden_layers=hidden_layers[:-1], **kwargs)
        self.ortho_init = ortho_init
        self.reset_parameters()

    def reset_parameters(self):
        if self.ortho_init:
            self.apply(partial(weight_init, gain=float(self.ortho_init)))  # use the fact that True converts to 1.0

    def forward(self, obs):
        return self.mlp(obs)


class MLPPredictor(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        hidden_layers: Iterable[int] = (256, 256),
        ortho_init: bool = False,
        **kwargs,
    ):
        assert isinstance(observation_space, gym.spaces.Box) and len(observation_space.shape) == 1
        assert len(hidden_layers) > 1, "Must have at least one hidden layer for a shared MLP Extractor"
        self.mlp = MLP(observation_space.shape[0], 1, hidden_layers=hidden_layers, **kwargs)
        self.ortho_init = ortho_init
        self.reset_parameters()

    def reset_parameters(self):
        if self.ortho_init:
            self.apply(partial(weight_init, gain=float(self.ortho_init)))  # use the fact that True converts to 1.0

    def forward(self, obs):
        return self.mlp(obs).squeeze(-1)


class MLPValue(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        ortho_init: bool = False,
        output_gain: Optional[float] = None,
        ensemble_size: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        assert isinstance(observation_space, gym.spaces.Box) and len(observation_space.shape) == 1

        self.ensemble_size = ensemble_size
        if self.ensemble_size > 1:
            self.mlp = EnsembleMLP(observation_space.shape[0], 1, ensemble_size=ensemble_size, **kwargs)
        else:
            self.mlp = MLP(observation_space.shape[0], 1, **kwargs)
        self.ortho_init = ortho_init
        self.output_gain = output_gain
        self.reset_parameters()

    def reset_parameters(self):
        if self.ortho_init:
            self.apply(partial(weight_init, gain=float(self.ortho_init)))  # use the fact that True converts to 1.0
            if self.output_gain is not None:
                self.mlp.last_layer.apply(partial(weight_init, gain=self.output_gain))

    def forward(self, obs):
        v = self.mlp(obs).squeeze(-1)  # Remove the last dim
        if self.ensemble_size == 1:
            v = v.unsqueeze(0)  # add in the ensemble dim
        return v


class ContinuousMLPCritic(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        ensemble_size: int = 2,
        ortho_init: bool = False,
        output_gain: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()
        assert isinstance(observation_space, gym.spaces.Box) and len(observation_space.shape) == 1
        self.ensemble_size = ensemble_size
        input_dim = observation_space.shape[0] + action_space.shape[0]
        if self.ensemble_size > 1:
            self.q = EnsembleMLP(input_dim, 1, ensemble_size=ensemble_size, **kwargs)
        else:
            self.q = MLP(input_dim, 1, **kwargs)

        self.ortho_init = ortho_init
        self.output_gain = output_gain
        self.reset_parameters()

    def reset_parameters(self):
        if self.ortho_init:
            self.apply(partial(weight_init, gain=float(self.ortho_init)))  # use the fact that True converts to 1.0
            if self.output_gain is not None:
                self.mlp.last_layer.apply(partial(weight_init, gain=self.output_gain))

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat((obs, action), dim=-1)
        q = self.q(x).squeeze(-1)  # Remove the last dim
        if self.ensemble_size == 1:
            q = q.unsqueeze(0)  # add in the ensemble dim
        return q


class DiscreteMLPCritic(nn.Module):
    """
    Q function for discrete Q learning. Currently doesn't support ensembles.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        ortho_init: bool = False,
        output_gain: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()
        self.q = MLP(observation_space.shape[0], action_space.n, **kwargs)
        self.ortho_init = ortho_init
        self.output_gain = output_gain
        self.reset_parameters()

    def reset_parameters(self):
        if self.ortho_init:
            self.apply(partial(weight_init, gain=float(self.ortho_init)))  # use the fact that True converts to 1.0
            if self.output_gain is not None:
                self.mlp.last_layer.apply(partial(weight_init, gain=self.output_gain))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.q(obs)


class ContinuousMLPActor(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        ortho_init: bool = False,
        output_gain: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()
        assert isinstance(observation_space, gym.spaces.Box) and len(observation_space.shape) == 1

        self.mlp = MLP(observation_space.shape[0], action_space.shape[0], **kwargs)
        self.ortho_init = ortho_init
        self.output_gain = output_gain
        self.reset_parameters()

    def reset_parameters(self):
        if self.ortho_init:
            self.apply(partial(weight_init, gain=float(self.ortho_init)))  # use the fact that True converts to 1.0
            if self.output_gain is not None:
                self.mlp.last_layer.apply(partial(weight_init, gain=self.output_gain))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)


class SquashedNormal(D.TransformedDistribution):
    def __init__(self, loc: torch.Tensor, scale: torch.Tensor) -> None:
        self._loc = loc
        self.scale = scale
        self.base_dist = D.Normal(loc, scale)
        transforms = [D.transforms.TanhTransform(cache_size=1)]
        super().__init__(self.base_dist, transforms)

    @property
    def loc(self) -> torch.Tensor:
        loc = self._loc
        for transform in self.transforms:
            loc = transform(loc)
        return loc


class DiagonalGaussianMLPActor(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        ortho_init: bool = False,
        output_gain: Optional[float] = None,
        log_std_bounds: Iterable[int] = (-5, 2),
        state_dependent_log_std: bool = True,
        squash_normal: bool = True,
        log_std_tanh: bool = True,
        output_act: Optional[Type[nn.Module]] = nn.Tanh,
        **kwargs,
    ):
        super().__init__()
        assert isinstance(observation_space, gym.spaces.Box) and len(observation_space.shape) == 1

        self.state_dependent_log_std = state_dependent_log_std
        self.log_std_bounds = log_std_bounds
        self.squash_normal = squash_normal
        self.log_std_tanh = log_std_tanh
        self.output_act = nn.Identity() if output_act is None else output_act()

        # Perform checks to make sure arguments are consistent
        assert log_std_bounds is None or log_std_bounds[0] < log_std_bounds[1], "invalid log_std bounds"
        assert not (output_act is not None and squash_normal), "Cannot use output act and squash normal"

        if self.state_dependent_log_std:
            action_dim = 2 * action_space.shape[0]
        else:
            action_dim = action_space.shape[0]
            self.log_std = nn.Parameter(
                torch.zeros(action_space.shape[0]), requires_grad=True
            )  # initialize a single parameter vector

        self.mlp = MLP(observation_space.shape[0], action_dim, output_act=None, **kwargs)
        self.ortho_init = ortho_init
        self.output_gain = output_gain
        self.reset_parameters()

    def reset_parameters(self):
        if self.ortho_init:
            self.apply(partial(weight_init, gain=float(self.ortho_init)))  # use the fact that True converts to 1.0
            if self.output_gain is not None:
                self.mlp.last_layer.apply(partial(weight_init, gain=self.output_gain))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.state_dependent_log_std:
            mu, log_std = self.mlp(obs).chunk(2, dim=-1)
        else:
            mu, log_std = self.mlp(obs), self.log_std
        mu = self.output_act(mu)
        if self.log_std_bounds is not None:
            if self.log_std_tanh:
                log_std = torch.tanh(log_std)
                log_std_min, log_std_max = self.log_std_bounds
                log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
            else:
                log_std = torch.clamp(log_std, *self.log_std_bounds)

        dist_class = SquashedNormal if self.squash_normal else D.Normal
        dist = dist_class(mu, log_std.exp())
        assert len(mu.shape) > 1, "Must used batched."
        dist = D.Independent(dist, 1)  # Combine logprob on last dim.
        return dist


class GaussianMixtureMLPActor(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        ortho_init: bool = False,
        output_gain: Optional[float] = None,
        num_modes: int = 5,
        log_std_bounds: Iterable[int] = (-5, 2),
        state_dependent_log_std: bool = True,
        squash_normal: bool = True,
        log_std_tanh: bool = True,
        output_act: Optional[Type[nn.Module]] = None,
        **kwargs,
    ):
        super().__init__()
        assert isinstance(observation_space, gym.spaces.Box) and len(observation_space.shape) == 1
        # Perform checks to make sure arguments are consistent
        assert log_std_bounds is None or log_std_bounds[0] < log_std_bounds[1], "invalid log_std bounds"
        assert not (output_act is not None and squash_normal), "Cannot use output act and squash normal"

        self.num_modes = num_modes
        self.act_dim = action_space.shape[0]

        self.state_dependent_log_std = state_dependent_log_std
        self.log_std_bounds = log_std_bounds
        self.squash_normal = squash_normal
        self.log_std_tanh = log_std_tanh
        self.output_act = nn.Identity() if output_act is None else output_act()

        output_dim = self.num_modes * self.act_dim

        if self.state_dependent_log_std:
            output_dim = 2 * output_dim  # Double dim
        else:
            # initialize a single parameter vector
            self.log_std = nn.Parameter(torch.zeros(output_dim), requires_grad=True)

        # Add dim for the logits.
        output_dim += self.num_modes

        self.mlp = MLP(observation_space.shape[0], output_dim, output_act=None, **kwargs)
        self.ortho_init = ortho_init
        self.output_gain = output_gain
        self.reset_parameters()

    def reset_parameters(self):
        if self.ortho_init:
            self.apply(partial(weight_init, gain=float(self.ortho_init)))  # use the fact that True converts to 1.0
            if self.output_gain is not None:
                self.mlp.last_layer.apply(partial(weight_init, gain=self.output_gain))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.mlp(obs)
        # Split on the last dim to remove the logits.
        x, logits = torch.split(x, (x.shape[-1] - self.num_modes, self.num_modes), dim=-1)
        if self.state_dependent_log_std:
            mu, log_std = x.chunk(2, dim=-1)
        else:
            mu, log_std = x, self.log_std
        mu = self.output_act(mu)
        if self.log_std_bounds is not None:
            if self.log_std_tanh:
                log_std = torch.tanh(log_std)
                log_std_min, log_std_max = self.log_std_bounds
                log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
            else:
                log_std = torch.clamp(log_std, *self.log_std_bounds)
        # Reshape the logits into the correct size.
        assert len(mu.shape) > 1, "Must used batched."
        mixture_shape = mu.shape[:-1] + (self.num_modes, self.act_dim)
        mu = mu.view(*mixture_shape)
        log_std = log_std.view(*mixture_shape)

        component_dist_class = SquashedNormal if self.squash_normal else D.Normal
        component_dist = component_dist_class(mu, log_std.exp())
        component_dist = D.Independent(component_dist, 1)

        cat_dist = D.Categorical(logits=logits)

        return D.MixtureSameFamily(mixture_distribution=cat_dist, component_distribution=component_dist)
