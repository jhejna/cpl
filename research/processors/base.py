"""
Processors are designed as ways of manipulating entire batches of tensors at once to prepare them for the network.
Examples are as follows:
1. Normalization
2. Image Augmentations applied on the entire batch at once.
"""
from typing import Any, Dict, List, Optional, Tuple

import gym
import torch

import research


class Processor(torch.nn.Module):
    """
    This is the base processor class. All processors should inherit from it.
    """

    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        super().__init__()
        self.training = True
        self._observation_space = observation_space
        self._action_space = action_space

    def unprocess(self, batch: Any) -> Any:
        raise NotImplementedError

    @property
    def supports_gpu(self):
        return True

    @property
    def observation_space(self):
        """
        Outputs the observation space for the network
        Can be overrided if processor changes this space.
        """
        return self._observation_space

    @property
    def action_space(self):
        """
        Outputs the action space for the network
        Can be overrided if processor changes this space.
        """
        return self._action_space


class Identity(Processor):
    """
    This processor just performs the identity operation
    """

    def forward(self, batch: Any) -> Any:
        return batch

    def unprocess(self, batch: Any) -> Any:
        return batch


class Compose(Processor):
    """
    This Processor Composes multiple processors
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        processors: List[Tuple[str, Optional[Dict]]] = (("Identity", None),),
    ):
        super().__init__(observation_space, action_space)
        created_processors = []
        current_observation_space, current_action_space = observation_space, action_space
        for processor_class, processor_kwargs in processors:
            processor_class = vars(research.processors)[processor_class]
            processor_kwargs = {} if processor_kwargs is None else processor_kwargs
            processor = processor_class(current_observation_space, current_action_space, **processor_kwargs)
            created_processors.append(processor)
            current_observation_space, current_action_space = processor.observation_space, processor.action_space
        self.processors = torch.nn.ModuleList(created_processors)

    @property
    def observation_space(self):
        # Return the space of the last processor
        return self.processors[-1].observation_space

    @property
    def action_space(self):
        # Return the space of the last processor
        return self.processors[-1].action_space

    @property
    def supports_gpu(self):
        return all([processor.supports_gpu for processor in self.processors])

    def forward(self, batch: Any) -> Any:
        for processor in self.processors:
            batch = processor(batch)
        return batch

    def unprocess(self, batch: Any) -> Any:
        for processor in reversed(self.processors):
            batch = processor.unprocess(batch)
        return batch
