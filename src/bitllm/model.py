import torch
from torch import nn


class DimensionError(Exception):
    pass


class BitLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor):
        pass

    def _compute_mean_of_weights(self, w: torch.Tensor) -> torch.Tensor:
        if not isinstance(w, torch.Tensor):
            raise TypeError("Expecting a torch tensor as input.")

        if (tensor_shape := len(w.size())) != 2:
            raise DimensionError(f"Expecting a n x m input. Got input shape {tensor_shape}")

        return w.mean()

    def _demean_weights(self, w: torch.Tensor) -> torch.Tensor:
        alpha = self._compute_mean_of_weights(w)
        return w - alpha

    def _get_sign_of_weights(self, w: torch.Tensor) -> torch.Tensor:
        if not isinstance(w, torch.Tensor):
            raise TypeError("Expecting a torch tensor as input.")

        if (tensor_shape := len(w.size())) != 2:
            raise DimensionError(f"Expecting a n x m input. Got input shape {tensor_shape}")

        sign_w = torch.sign(w)
        sign_w = sign_w.to(torch.int8)
        return sign_w
