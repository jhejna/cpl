import functools
from typing import Dict, Optional, Tuple, Union

import gym
import numpy as np
import torch
import torchvision
from torch.nn import functional as F

from .base import Processor


def is_image_space(space):
    shape = space.shape
    is_image_space = (len(shape) == 3 or len(shape) == 4) and space.dtype == np.uint8
    return is_image_space


def modify_space_hw(space, h, w):
    if isinstance(space, gym.spaces.Box) and is_image_space(space):
        shape = list(space.shape)
        shape[-2] = h
        shape[-1] = w
        return gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
    elif isinstance(space, gym.spaces.Dict):
        return gym.spaces.Dict({k: modify_space_hw(v, h, w) for k, v in space.items()})
    else:
        return space


class RandomCrop(Processor):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        size: Optional[Tuple[int, int]] = None,
        pad: Union[int, Tuple[int, int]] = 4,
        consistent: bool = True,
    ) -> None:
        super().__init__(observation_space, action_space)
        self.consistent = consistent

        # Get the image keys and sequence lengths
        if isinstance(observation_space, gym.spaces.Box):
            assert is_image_space(observation_space)
            self.is_sequence = len(observation_space.shape) == 4
            self.in_h, self.in_w = observation_space.shape[-2], observation_space.shape[-1]
            self.image_keys = None
        elif isinstance(observation_space, gym.spaces.Dict):
            image_keys = []
            sequence = []
            hs, ws = [], []
            for k, v in observation_space.items():
                if is_image_space(v):
                    image_keys.append(k)
                    if len(v.shape) == 4:
                        sequence.append(v.shape[0])  # Append the sequence dim
                    else:
                        sequence.append(0)
                    ws.append(v.shape[-1])
                    hs.append(v.shape[-2])
            assert all(sequence) or (not any(sequence)), "All image keys must be sequence or not"
            assert all([h == hs[0] for h in hs])
            assert all([w == ws[0] for w in ws])
            self.in_h, self.in_w = hs[0], ws[0]
            self.is_sequence = sequence[0]
            self.image_keys = image_keys
        else:
            raise ValueError("Invalid observation space specified")

        # Save output sizes
        if size is None:
            self.out_h, self.out_w = self.in_h, self.in_w
        else:
            self.out_h, self.out_w = size
        assert self.out_h <= self.in_h and self.out_w <= self.in_w

        self.pad = (pad, pad) if isinstance(pad, int) else pad
        self.padding = [self.pad[0], self.pad[0], self.pad[1], self.pad[1]]
        self.do_pad = self.pad[0] > 0 or self.pad[1] > 0

        # Save intermediate sizes
        self.middle_h, self.middle_w = self.in_h + 2 * self.pad[0], self.in_w + 2 * self.pad[1]

        self.is_square = self.in_h == self.in_w
        if self.is_square:
            assert self.out_h == self.out_w, "Must use square output on square images for acceleration"
            assert self.pad[0] == self.pad[1], "Must use uniform pad with square images"

        eps_h = 1.0 / (self.middle_h)
        eps_w = 1.0 / (self.middle_w)

        grid_h = torch.linspace(-1.0 + eps_h, 1.0 - eps_h, self.middle_h, dtype=torch.float)[: self.out_h]
        grid_h = grid_h.unsqueeze(1).repeat(1, self.out_w)
        grid_w = torch.linspace(-1.0 + eps_w, 1.0 - eps_w, self.middle_w, dtype=torch.float)[: self.out_w]
        grid_w = grid_w.unsqueeze(0).repeat(self.out_h, 1)
        base_grid = torch.stack((grid_w, grid_h), dim=-1).unsqueeze(0)  # Shape (1, out_h, out_w, 2)

        self.register_buffer("base_grid", base_grid, persistent=False)  # Do note save the grid in state_dict

        # Now set the eval op
        if self.out_h == self.in_h and self.out_w == self.in_w:
            self.eval_op = None
        else:
            self.eval_op = functools.partial(
                torchvision.transforms.functional.center_crop, output_size=(self.out_h, self.out_w)
            )

    @property
    def observation_space(self):
        return modify_space_hw(self._observation_space, self.out_h, self.out_w)

    def _aug(self, x: torch.Tensor) -> torch.Tensor:
        size = x.size()
        assert len(size) == 4, "_aug supports images of shape (b, c, h, w)"
        b = size[0]
        # Determine if we should pad
        if self.do_pad:
            x = F.pad(x, self.padding, "replicate")

        if self.is_square:
            # offsets are computed in the pad and subsample size
            offsets = (
                torch.randint(0, self.middle_h - self.out_h + 1, size=(b, 1, 1, 2), device=x.device, dtype=torch.float)
                * 2.0
                / (self.middle_h)
            )
        else:
            # We need to compute individual h and w offsets.
            h_offsets = (
                torch.randint(0, self.middle_h - self.out_h + 1, size=(b, 1, 1), device=x.device, dtype=torch.float)
                * 2.0
                / (self.middle_h)
            )
            w_offsets = (
                torch.randint(0, self.middle_w - self.out_w + 1, size=(b, 1, 1), device=x.device, dtype=torch.float)
                * 2.0
                / (self.middle_w)
            )
            offsets = torch.stack((w_offsets, h_offsets), dim=-1)

        grid = self.base_grid + offsets

        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)

    def forward(self, batch: Dict) -> Dict:
        op = self._aug
        if not self.training:
            if self.eval_op is None:
                return batch
            else:
                op = self.eval_op

        # Images are assumed to be of shape (B, S, C, H, W) or (B, C, H, W) if there is no sequence dimension
        images = []
        split = []
        for k in ("obs", "next_obs", "init_obs", "obs_1", "obs_2"):
            if k in batch:
                if self.image_keys is None:
                    images.append(batch[k])
                    split.append(batch[k].shape[1])
                else:
                    images.extend([batch[k][img_key] for img_key in self.image_keys])
                    split.extend([batch[k][img_key].shape[1] for img_key in self.image_keys])

        is_sequence = self.is_sequence or len(images[0].shape) > 4  # See if we have a sequence dimension
        with torch.no_grad():
            images = torch.cat(images, dim=1 if self.consistent else 0)  # This is either the seq dim or channel dim.
            if is_sequence:
                n, s, c, h, w = images.size()
                images = images.view(n, s * c, h, w)  # Apply same augmentations across sequence.
            images = op(images.float())  # Apply the same augmentation to each data pt.
            if is_sequence:
                images = images.view(n, s, c, h, w)
            # Split according to the dimension 1 splits
            images = torch.split(images, split, dim=1 if self.consistent else 0)

        # Iterate over everything in the same order and overwrite in the batch
        i = 0
        for k in ("obs", "next_obs", "init_obs", "obs_1", "obs_2"):
            if k in batch:
                if self.image_keys is None:
                    batch[k] = images[i]
                    i += 1
                else:
                    for img_key in self.image_keys:
                        batch[k][img_key] = images[i]
                        i += 1
        assert i == len(images), "Did not write batch all augmented images."
        return batch
