from typing import List

import gym
import numpy as np
import torch
from torch import nn

from .mlp import MLP, EnsembleMLP


def drqv2_weight_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class DrQv2Encoder(nn.Module):
    def __init__(self, observation_space: gym.Space, action_space: gym.Space) -> None:
        super().__init__()
        if len(observation_space.shape) == 4:
            s, c, h, w = observation_space.shape
            channels = s * c
        elif len(observation_space.shape) == 3:
            c, h, w = observation_space.shape
            channels = c
        else:
            raise ValueError("Invalid observation space for DRQV2 Image encoder.")
        self.convnet = nn.Sequential(
            nn.Conv2d(channels, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.reset_parameters()

        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]) / 255.0 - 0.5
            self.repr_dim = self.convnet(sample).shape[1]

    def reset_parameters(self):
        self.apply(drqv2_weight_init)

    @property
    def output_space(self) -> gym.Space:
        return gym.spaces.Box(shape=(self.repr_dim,), low=-np.inf, high=np.inf, dtype=np.float32)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        is_seq = len(obs.shape) == 5
        if is_seq:
            b, s, c, h, w = obs.shape
            obs = obs.view(b * s, c, h, w)
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        if is_seq:
            h = h.view(b, s, self.repr_dim)
        return h


class DrQv2Critic(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        feature_dim: int = 50,
        hidden_layers: List[int] = (1024, 1024),
        ensemble_size: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(observation_space.shape[0], feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
        )
        self.ensemble_size = ensemble_size
        input_dim = feature_dim + action_space.shape[0]
        if self.ensemble_size > 1:
            self.mlp = EnsembleMLP(input_dim, 1, ensemble_size=ensemble_size, hidden_layers=hidden_layers, **kwargs)
        else:
            self.mlp = MLP(input_dim, 1, hidden_layers=hidden_layers, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(drqv2_weight_init)

    def forward(self, obs, action):
        x = self.trunk(obs)
        x = torch.cat((x, action), dim=-1)
        q = self.mlp(x).squeeze(-1)  # Remove the last dim
        if self.ensemble_size == 1:
            q = q.unsqueeze(0)  # add in the ensemble dim
        return q


class DrQv2Value(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        feature_dim: int = 50,
        hidden_layers: List[int] = (1024, 1024),
        ensemble_size: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(observation_space.shape[0], feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
        )
        self.ensemble_size = ensemble_size
        if self.ensemble_size > 1:
            self.mlp = EnsembleMLP(feature_dim, 1, ensemble_size=ensemble_size, hidden_layers=hidden_layers, **kwargs)
        else:
            self.mlp = MLP(feature_dim, 1, hidden_layers=hidden_layers, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(drqv2_weight_init)

    def forward(self, obs):
        v = self.trunk(obs)
        v = self.mlp(v).squeeze(-1)  # Remove the last dim
        if self.ensemble_size == 1:
            v = v.unsqueeze(0)  # add in the ensemble dim
        return v


class DrQv2Actor(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        feature_dim: int = 50,
        hidden_layers: List[int] = (1024, 1024),
        **kwargs,
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(observation_space.shape[0], feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
        )
        self.mlp = MLP(feature_dim, action_space.shape[0], hidden_layers=hidden_layers, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(drqv2_weight_init)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.trunk(obs)
        return self.mlp(x)


class DrQv2Reward(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        feature_dim: int = 50,
        hidden_layers: List[int] = (1024, 1024),
        ensemble_size: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.encoder = DrQv2Encoder(observation_space, action_space)

        self.trunk = nn.Sequential(nn.Linear(self.encoder.repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())
        self.ensemble_size = ensemble_size
        input_dim = feature_dim + action_space.shape[0]
        if self.ensemble_size > 1:
            self.mlp = EnsembleMLP(input_dim, 1, ensemble_size=ensemble_size, hidden_layers=hidden_layers, **kwargs)
        else:
            self.mlp = MLP(input_dim, 1, hidden_layers=hidden_layers, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(drqv2_weight_init)

    def forward(self, obs, action):
        x = self.encoder(obs)
        x = self.trunk(x)
        x = torch.cat((x, action), dim=-1)
        q = self.mlp(x).squeeze(-1)  # Remove the last dim
        if self.ensemble_size == 1:
            q = q.unsqueeze(0)  # add in the ensemble dim
        return q
