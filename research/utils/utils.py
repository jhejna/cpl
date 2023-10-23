from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import h5py
import numpy as np
import torch


def to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return [to_device(v, device) for v in batch]
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, (int, float, type(None))):
        return batch
    else:
        raise ValueError("Unsupported type passed to `to_device`")


def to_tensor(batch: Any) -> Any:
    if isinstance(batch, dict):
        return {k: to_tensor(v) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return [to_tensor(v) for v in batch]
    elif isinstance(batch, np.ndarray):
        # Special case to handle float64 -- which we never want to use with pytorch
        if batch.dtype == np.float64:
            batch = batch.astype(np.float32)
        return torch.from_numpy(batch)
    elif isinstance(batch, (int, float, type(None))):
        return batch
    else:
        raise ValueError("Unsupported type passed to `to_tensor`")


def to_np(batch: Any) -> Any:
    if isinstance(batch, dict):
        return {k: to_np(v) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return [to_np(v) for v in batch]
    elif isinstance(batch, torch.Tensor):
        return batch.detach().cpu().numpy()
    else:
        raise ValueError("Unsupported type passed to `to_np`")


def remove_float64(batch: Any):
    if isinstance(batch, dict):
        return {k: remove_float64(v) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return [remove_float64(v) for v in batch]
    elif isinstance(batch, np.ndarray):
        if batch.dtype == np.float64:
            return batch.astype(np.float32)
    elif isinstance(batch, torch.Tensor):
        if batch.dtype == torch.double:
            return batch.float()
    else:
        raise ValueError("Unsupported type passed to `remove_float64`")
    return batch


def unsqueeze(batch: Any, dim: int) -> Any:
    if isinstance(batch, dict):
        return {k: unsqueeze(v, dim) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return [unsqueeze(v, dim) for v in batch]
    elif isinstance(batch, np.ndarray):
        return np.expand_dims(batch, dim)
    elif isinstance(batch, torch.Tensor):
        return batch.unsqueeze(dim)
    elif isinstance(batch, (int, float, np.generic)):
        return np.array([batch])
    else:
        raise ValueError("Unsupported type passed to `unsqueeze`")


def squeeze(batch: Any, dim: int) -> Any:
    if isinstance(batch, dict):
        return {k: squeeze(v, dim) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return [squeeze(v, dim) for v in batch]
    elif isinstance(batch, np.ndarray):
        return np.squeeze(batch, axis=dim)
    elif isinstance(batch, torch.Tensor):
        return batch.squeeze(dim)
    else:
        raise ValueError("Unsupported type passed to `squeeze`")


def get_from_batch(batch: Any, start: Union[int, np.ndarray, torch.Tensor], end: Optional[int] = None) -> Any:
    if isinstance(batch, (dict, h5py.Group)):
        return {k: get_from_batch(v, start, end=end) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return [get_from_batch(v, start, end=end) for v in batch]
    elif isinstance(batch, (np.ndarray, torch.Tensor, h5py.Dataset)):
        if end is None:
            return batch[start]
        else:
            return batch[start:end]
    else:
        raise ValueError("Unsupported type passed to `get_from_batch`")


def set_in_batch(batch: Any, value: Any, start: int, end: Optional[int] = None) -> None:
    if isinstance(batch, dict):
        for k, v in batch.items():
            set_in_batch(v, value[k], start, end=end)
    elif isinstance(batch, (list, tuple)):
        for v in batch:
            set_in_batch(v, value, start, end=end)
    elif isinstance(batch, np.ndarray) or isinstance(batch, torch.Tensor):
        if end is None:
            batch[start] = value
        else:
            batch[start:end] = value
    else:
        raise ValueError("Unsupported type passed to `set_in_batch`")


def batch_copy(batch: Any) -> Any:
    if isinstance(batch, dict):
        return {k: batch_copy(v) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return [batch_copy(v) for v in batch]
    elif isinstance(batch, np.ndarray):
        return batch.copy()
    elif isinstance(batch, torch.Tensor):
        return batch.clone()
    # Note that if we have scalars etc. we just return the value, thus no ending check.
    return batch


def space_copy(space: gym.Space):
    # A custom method for copying gym spaces.
    # this is because numpy 1.24.0 changed how random states are stored, making them
    # unserializable by the copy library.
    if isinstance(space, gym.spaces.Dict):
        return gym.spaces.Dict({k: space_copy(v) for k, v in space.items()})
    elif isinstance(space, gym.spaces.Box):
        return gym.spaces.Box(low=space.low, high=space.high, dtype=space.dtype)
    elif isinstance(space, gym.spaces.Discrete):
        return gym.spaces.Discrete(n=space.n)
    else:
        raise ValueError("Invalid space passed to `space_copy`.")


def contains_tensors(batch: Any) -> bool:
    if isinstance(batch, dict):
        return any([contains_tensors(v) for v in batch.values()])
    if isinstance(batch, list):
        return any([contains_tensors(v) for v in batch])
    elif isinstance(batch, torch.Tensor):
        return True
    else:
        return False


def get_device(batch: Any) -> Optional[torch.device]:
    if isinstance(batch, dict):
        return get_device(list(batch.values()))
    elif isinstance(batch, list):
        devices = [get_device(d) for d in batch]
        for d in devices:
            if d is not None:
                return d
        else:
            return None
    elif isinstance(batch, torch.Tensor):
        return batch.device
    else:
        return None


def concatenate(*args, dim: int = 0):
    assert all([isinstance(arg, type(args[0])) for arg in args]), "Must concatenate tensors of the same type"
    if isinstance(args[0], dict):
        return {k: concatenate(*[arg[k] for arg in args], dim=dim) for k in args[0].keys()}
    elif isinstance(args[0], list) or isinstance(args[0], tuple):
        return [concatenate(*[arg[i] for arg in args], dim=dim) for i in range(len(args[0]))]
    elif isinstance(args[0], np.ndarray):
        return np.concatenate(args, axis=dim)
    elif isinstance(args[0], torch.Tensor):
        return torch.concatenate(args, dim=dim)
    else:
        raise ValueError("Unsupported type passed to `concatenate`")


def append(lst, item):
    # This takes in a nested list structure and appends everything from item to the nested list structure.
    # It will require lst to have the complete set of keys -- if keys are in item but not in lst,
    # they will not be appended.
    if isinstance(lst, dict):
        assert isinstance(item, dict)
        for k in lst.keys():
            append(lst[k], item[k])
    else:
        lst.append(item)


def extend(lst1, lst2):
    # This takes in a nested list structure and appends everything from item to the nested list structure.
    # It will require lst to have the complete set of keys -- if keys are in item but not in lst,
    # they will not be extended
    if isinstance(lst1, dict):
        assert isinstance(lst2, dict)
        for k in lst1.keys():
            extend(lst1[k], lst2[k])
    else:
        lst1.extend(lst2)


class PrintNode(torch.nn.Module):
    def __init__(self, name: str = ""):
        super().__init__()
        self.name = name

    def forward(self, x: Any) -> Any:
        print(self.name, x.shape)
        return x


def np_dataset_alloc(
    space: gym.Space, capacity: int, begin_pad: Tuple[int] = tuple(), end_pad: Tuple[int] = tuple()
) -> np.ndarray:
    if isinstance(space, (dict, gym.spaces.Dict)):
        return {k: np_dataset_alloc(v, capacity, begin_pad=begin_pad, end_pad=end_pad) for k, v in space.items()}
    elif isinstance(space, bool):
        return np.empty((capacity, *begin_pad, *end_pad), dtype=np.bool_)
    elif isinstance(space, (gym.spaces.Box, np.ndarray)):
        dtype = np.float32 if space.dtype == np.float64 else space.dtype
        return np.empty((capacity, *begin_pad, *space.shape, *end_pad), dtype=dtype)
    elif isinstance(space, gym.spaces.Discrete) or isinstance(space, (int, np.int64)):
        return np.empty((capacity, *begin_pad, *end_pad), dtype=np.int64)
    elif isinstance(space, float) or isinstance(space, np.float32):
        return np.empty((capacity, *begin_pad, *end_pad), dtype=np.float32)
    else:
        raise ValueError("Invalid space provided to `np_dataset_alloc`")


def np_bytes_per_instance(space: gym.Space) -> int:
    if isinstance(space, gym.spaces.Dict):
        return sum([np_bytes_per_instance(v) for k, v in space.items()])
    elif isinstance(space, bool):
        return np.dtype(np.bool_).itemsize
    elif isinstance(space, (gym.spaces.Box, np.ndarray)):
        dtype = np.float32 if space.dtype == np.float64 else space.dtype
        return np.dtype(dtype).itemsize * np.prod(space.shape)
    elif isinstance(space, gym.spaces.Discrete) or isinstance(space, (int, np.int64)):
        return np.dtype(np.int64).itemsize
    elif isinstance(space, float) or isinstance(space, np.float32):
        return np.dtype(np.float32).itemsize
    else:
        raise ValueError("Invalid space provided to `np_bytes_per_instance`")


def _flatten_dict_helper(flat_dict: Dict, value: Any, prefix: str, separator: str = ".") -> None:
    if isinstance(value, (dict, gym.spaces.Dict)):
        for k in value.keys():
            assert isinstance(k, str), "Can only flatten dicts with str keys"
            _flatten_dict_helper(flat_dict, value[k], prefix + separator + k, separator=separator)
    else:
        flat_dict[prefix[1:]] = value


def flatten_dict(d: Dict, separator: str = ".") -> Dict:
    flat_dict = dict()
    _flatten_dict_helper(flat_dict, d, "", separator=separator)
    return flat_dict


def nest_dict(d: Dict, separator: str = ".") -> Dict:
    nested_d = dict()
    for key in d.keys():
        key_parts = key.split(separator)
        current_d = nested_d
        while len(key_parts) > 1:
            if key_parts[0] not in current_d:
                current_d[key_parts[0]] = dict()
            current_d = current_d[key_parts[0]]
            key_parts.pop(0)
        current_d[key_parts[0]] = d[key]  # Set the value
    return nested_d


def fetch_from_dict(d: Dict, keys: Union[str, List, Tuple], separator=".") -> List[Any]:
    """
    inputs:
        d: a nested dictionary datastrucutre
        keys: a list of string keys, with '.' separating nested items.
    """
    outputs = []
    if not isinstance(keys, list) and not isinstance(keys, tuple):
        keys = [keys]
    for key in keys:
        key_parts = key.split(separator)
        current_dict = d
        while len(key_parts) > 1:
            current_dict = current_dict[key_parts[0]]
            key_parts.pop(0)
        outputs.append(current_dict[key_parts[0]])
    if len(outputs) == 1:
        outputs = outputs[0]
    return outputs


def create_optim_groups(params, kwargs):
    # create optim groups. Any parameters that is 2D or higher will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    params = list(params)
    if kwargs.get("weight_decay", 0.0) == 0.0:
        group = {"params": [p for p in params if p.requires_grad]}
        group.update(kwargs)
        return (group,)
    else:
        # We have a decay group
        decay_group = {"params": [p for p in params if p.dim() >= 2 and p.requires_grad]}
        no_decay_group = {"params": [p for p in params if p.dim() < 2 and p.requires_grad]}
        decay_group.update(kwargs)
        no_decay_group.update(kwargs)
        no_decay_group["weight_decay"] = 0.0
        return (decay_group, no_decay_group)
