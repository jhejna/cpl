import math
from typing import List, Optional, Tuple, Type, Union

import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: List[int] = (256, 256),
        act: nn.Module = nn.ReLU,
        dropout: float = 0.0,
        normalization: Optional[Type[nn.Module]] = None,
        output_act: Optional[Type[nn.Module]] = None,
    ):
        super().__init__()
        net = []
        last_dim = input_dim
        for dim in hidden_layers:
            net.append(nn.Linear(last_dim, dim))
            if dropout > 0.0:
                net.append(nn.Dropout(dropout))
            if normalization is not None:
                net.append(normalization(dim))
            net.append(act())
            last_dim = dim
        net.append(nn.Linear(last_dim, output_dim))
        if output_act is not None:
            net.append(output_act())
        self.net = nn.Sequential(*net)
        self._has_output_act = False if output_act is None else True

    @property
    def last_layer(self) -> nn.Module:
        if self._has_output_act:
            return self.net[-2]
        else:
            return self.net[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LinearEnsemble(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        ensemble_size: int = 3,
        bias: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        An Ensemble linear layer.
        For inputs of shape (B, H) will return (E, B, H) where E is the ensemble size
        See https://github.com/pytorch/pytorch/issues/54147
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.empty((ensemble_size, in_features, out_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty((ensemble_size, 1, out_features), **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # The default torch init for Linear is a complete mess
        # https://github.com/pytorch/pytorch/issues/57109
        # If we use the same init, we will end up scaling incorrectly
        # 1. Compute the fan in of the 2D tensor = dim 1 of 2D matrix (0 index)
        # 2. Comptue the gain with param=math.sqrt(5.0)
        #   This returns math.sqrt(2.0 / 6.0) = sqrt(1/3)
        # 3. Compute std = gain / math.sqrt(fan) = sqrt(1/3) / sqrt(in).
        # 4. Compute bound as math.sqrt(3.0) * std = 1 / in di
        std = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -std, std)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -std, std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 2:
            x = x.repeat(self.ensemble_size, 1, 1)
        elif len(x.shape) > 3:
            raise ValueError("LinearEnsemble layer does not support inputs with more than 3 dimensions.")
        return torch.baddbmm(self.bias, x, self.weight)

    def extra_repr(self) -> str:
        return "ensemble_size={}, in_features={}, out_features={}, bias={}".format(
            self.ensemble_size, self.in_features, self.out_features, self.bias is not None
        )


class LayerNormEnsemble(nn.Module):
    """
    This is a re-implementation of the Pytorch nn.LayerNorm module with suport for the Ensemble dim.
    We need this custom class since we need to normalize over normalize dims, but have multiple weight/bias
    parameters for the ensemble.

    """

    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape: int,
        ensemble_size: int = 3,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        assert isinstance(normalized_shape, int), "Currently EnsembleLayerNorm only supports final dim int shapes."
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.ensemble_size = ensemble_size
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty((self.ensemble_size, 1, *self.normalized_shape), **factory_kwargs))
            self.bias = nn.Parameter(torch.empty((self.ensemble_size, 1, *self.normalized_shape), **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 2:
            x = x.repeat(self.ensemble_size, 1, 1)
        elif len(x.shape) > 3:
            raise ValueError("LayerNormEnsemble layer does not support inputs with more than 3 dimensions.")
        x = F.layer_norm(x, self.normalized_shape, None, None, self.eps)  # (E, B, *normalized shape)
        if self.elementwise_affine:
            x = x * self.weight + self.bias
        return x

    def extra_repr(self) -> str:
        return "{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}".format(**self.__dict__)


class EnsembleMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        ensemble_size: int = 3,
        hidden_layers: List[int] = (256, 256),
        act: nn.Module = nn.ReLU,
        dropout: float = 0.0,
        normalization: Optional[Type[nn.Module]] = None,
        output_act: Optional[Type[nn.Module]] = None,
    ):
        """
        An ensemble MLP
        Returns values of shape (E, B, H) from input (B, H)
        All extra dimensions are moved to batch
        """
        super().__init__()
        # Change the normalization type to work over ensembles
        assert normalization is None or normalization is LayerNormEnsemble, "Ensemble only support EnsembleLayerNorm"
        net = []
        last_dim = input_dim
        for dim in hidden_layers:
            net.append(LinearEnsemble(last_dim, dim, ensemble_size=ensemble_size))
            if dropout > 0.0:
                net.append(nn.Dropout(dropout))
            if normalization is not None:
                net.append(normalization(dim, ensemble_size=ensemble_size))
            net.append(act())
            last_dim = dim
        net.append(LinearEnsemble(last_dim, output_dim, ensemble_size=ensemble_size))
        if output_act is not None:
            net.append(output_act())
        self.net = nn.Sequential(*net)
        self.input_dim, self.output_dim = input_dim, output_dim
        self.ensemble_size = ensemble_size
        self._has_output_act = False if output_act is None else True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The input to this network is assumed to be (....., input_dim)
        assert x.shape[-1] == self.input_dim
        batch_dims = x.size()[:-1]
        if len(batch_dims) > 1:
            x = x.view(-1, self.input_dim)
            x = self.net(x)
            output_shape = (self.ensemble_size, *batch_dims, self.output_dim)
            x = x.view(*output_shape)
        else:
            x = self.net(x)
        return x

    @property
    def last_layer(self) -> torch.Tensor:
        if self._has_output_act:
            return self.net[-2]
        else:
            return self.net[-1]
