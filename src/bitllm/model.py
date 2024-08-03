import torch
from torch import nn


class DimensionError(Exception):
    pass


def clip_activation(x: torch.Tensor, a: float, b: float) -> torch.Tensor:
    x[x > b] = b
    x[x < a] = a
    return x


def compute_infinity_norm(x: torch.Tensor) -> torch.Tensor:
    if len(x.size()) != 3:
        raise DimensionError("Expecting an activation of size Batch * Context Length * Model dim")
    return torch.amax(torch.abs(x), dim=(1, 2))


def _compute_quantization_range(b_bit_precision: int) -> torch.Tensor:
    return torch.tensor([-(2 ** (b_bit_precision - 1)), 2 ** (b_bit_precision - 1)])


# def quantize_activation(x: torch.Tensor, b_bit_precision: int = 8, floor: Optional[int] = None) -> torch.Tensor:
#     pass


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
