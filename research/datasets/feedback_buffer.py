import math
from typing import Optional

import gym
import numpy as np
import torch

from research.utils import utils


class FeedbackBuffer(torch.utils.data.IterableDataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        path: Optional[str] = None,
        discount: float = 0.99,
        action_eps: float = 1e-5,
        segment_length: Optional[int] = None,
        batch_size: Optional[int] = None,
        capacity: Optional[int] = None,
        mode: str = "comparison",
        label_key: str = "label",
        reward_scale: float = 1.0,
        reward_shift: float = 0.0,
    ):
        # assert mode in {"rank", "comparison", "score"}
        self.mode = mode
        self.label_key = label_key
        self.discount = discount
        self.batch_size = 1 if batch_size is None else batch_size
        self.segment_length = segment_length

        if mode.startswith("comparison") and capacity is not None:
            capacity = 2 * capacity

        assert path is not None, "Must provide dataset file."
        with open(path, "rb") as f:
            data = np.load(f)
            data = utils.nest_dict(data)
            assert self.label_key in data, "Key not found, valid keys:" + str(list(data.keys()))

            # Determine if we are dealing with an old format dataset
            # If so, convert to the new format by stacking.
            if "action_1" in data:
                label = data[self.label_key]
                data = {
                    "obs": utils.concatenate(data["obs_1"], data["obs_2"]),
                    "action": utils.concatenate(data["action_1"], data["action_2"]),
                    "reward": utils.concatenate(data["reward_1"], data["reward_2"]),
                }
                data[self.label_key] = utils.concatenate(1 - label, label)

            # If we are dealing with a new format dataset
            dataset_size = data["action"].shape[0]  # The number of segments in the dataset
            assert capacity is None or capacity <= dataset_size, "Capacity is set larger than dataset!"
            if capacity is not None and dataset_size > capacity:
                # Trim the dataset down
                data = utils.get_from_batch(data, 0, capacity)

        # preprocess the data
        data = utils.remove_float64(data)
        lim = 1 - action_eps
        data["action"] = np.clip(data["action"], a_min=-lim, a_max=lim)
        data["reward"] = reward_scale * data["reward"] + reward_shift

        # Save the data
        self.data = data

    def __len__(self):
        return self.data["action"].shape[0]

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        size = len(self)
        if self.mode.startswith("comparison"):
            size = size // 2  # Trim down

        chunk_size = size // num_workers
        my_inds = np.arange(chunk_size * worker_id, chunk_size * (worker_id + 1))
        idxs = np.random.permutation(my_inds)
        for i in range(math.ceil(len(idxs) / self.batch_size)):  # Need to use ceil to get all data points.
            if self.batch_size == 1:
                data_1_idxs = idxs[i]
            else:
                # Might be some overlap here but its probably OK.
                data_1_idxs = idxs[i * self.batch_size : min((i + 1) * self.batch_size, len(self))]

            if self.mode == "comparison":
                data_2_idxs = data_1_idxs + size
            elif self.mode == "rank":
                data_2_idxs = np.random.randint(0, size, size=data_1_idxs.shape)
            elif self.mode.startswith("comparison_"):
                max_gap = int(self.mode.split("_")[1])
                gap = np.random.randint(0, max_gap, size=data_1_idxs.shape)
                data_2_idxs = data_1_idxs + size - gap

            dataset_segment_length = self.data["action"].shape[1]
            if self.segment_length is not None:
                # Trim down the dataset based on segment lengths
                start_1 = np.random.randint(0, dataset_segment_length - self.segment_length)
                end_1 = start_1 + self.segment_length
                start_2 = np.random.randint(0, dataset_segment_length - self.segment_length)
                end_2 = start_2 + self.segment_length
            else:
                start_1, end_1 = 0, dataset_segment_length
                start_2, end_2 = 0, dataset_segment_length

            if self.mode == "score":
                # Return the score batch
                batch = {
                    "obs": self.data["obs"][data_1_idxs, start_1:end_1],
                    "action": self.data["action"][data_1_idxs, start_2:end_2],
                    "reward": self.data["reward"][data_1_idxs, start_1:end_1],
                    "score": self.data[self.label_key][data_1_idxs],
                }
                batch["discount"] = self.discount * np.ones_like(batch["reward"], dtype=np.float32)
            else:
                # Return batch with comparisons
                batch = {
                    "obs_1": self.data["obs"][data_1_idxs, start_1:end_1],
                    "obs_2": self.data["obs"][data_2_idxs, start_2:end_2],
                    "action_1": self.data["action"][data_1_idxs, start_1:end_1],
                    "action_2": self.data["action"][data_2_idxs, start_2:end_2],
                    "reward_1": self.data["reward"][data_1_idxs, start_1:end_1],
                    "reward_2": self.data["reward"][data_2_idxs, start_2:end_2],
                }

                hard_label = 1.0 * (self.data[self.label_key][data_1_idxs] < self.data[self.label_key][data_2_idxs])
                soft_label = 0.5 * (self.data[self.label_key][data_1_idxs] == self.data[self.label_key][data_2_idxs])
                batch["label"] = (hard_label + soft_label).astype(np.float32)
                batch["discount_1"] = self.discount * np.ones_like(batch["reward_1"], dtype=np.float32)
                batch["discount_2"] = self.discount * np.ones_like(batch["reward_2"], dtype=np.float32)

            yield batch
