#TODO: adjust dimensions to take into account batch size

from typing import Callable
from typing import List

import numpy as np
import pytest
import torch
from bitllm.model import BitLinear
from bitllm.model import DimensionError
from bitllm.model import _compute_quantization_range
from bitllm.model import clip_activation
from bitllm.model import compute_infinity_norm


@pytest.fixture
def bitlinear_layer() -> BitLinear:
    return BitLinear()


@pytest.fixture
def weigth_matrix() -> Callable:
    def _weight_matrix(n: int, m: int) -> torch.Tensor:
        return torch.rand(n, m)

    return _weight_matrix


@pytest.mark.parametrize("n,m", [(2, 5), (12, 34)])
def test__compute_mean_of_weights_n_x_m(bitlinear_layer, weigth_matrix, n, m):
    w = weigth_matrix(12, 6)
    n, m = w.size()
    average = w.sum() / (n * m)
    bitlinear = bitlinear_layer
    assert bitlinear._compute_mean_of_weights(w) == average


def test__demean_weights_has_0_mean(bitlinear_layer, weigth_matrix):
    demeaned_W = bitlinear_layer._demean_weights(weigth_matrix(3, 6))
    assert demeaned_W.mean().item() == pytest.approx(0, abs=1e-6)


@pytest.fixture(
    params=[
        torch.tensor([]),
        torch.tensor(2.0),
        torch.tensor([1.0]),
        torch.tensor([[[1.0, 2.0], [1.0, 2.0]], [[1.0, 2.0], [1.0, 2.0]]]),
    ]
)
def tensors_wrong_dims(request) -> List[torch.Tensor]:
    return request.param


def test__compute_mean_of_weights_wrong_dim(bitlinear_layer, tensors_wrong_dims):
    with pytest.raises(DimensionError):
        bitlinear_layer._compute_mean_of_weights(tensors_wrong_dims)


def test__demean_weights_wrong_dim(bitlinear_layer, tensors_wrong_dims):
    with pytest.raises(DimensionError):
        bitlinear_layer._demean_weights(tensors_wrong_dims)


def test__get_sign_of_weights_wrong_dim(bitlinear_layer, tensors_wrong_dims):
    with pytest.raises(DimensionError):
        bitlinear_layer._get_sign_of_weights(tensors_wrong_dims)


@pytest.fixture(params=[[1, 2, 3], np.array([[1.0, 2.0], [3.0, 4.0]]), "abc"])
def tensorts_wrong_types(request):
    return request.param


def test__demean_weights_with_wrong_type(bitlinear_layer, tensorts_wrong_types):
    with pytest.raises(TypeError):
        bitlinear_layer._demean_weights(tensorts_wrong_types)


def test__compute_mean_of_weights_with_wrong_type(bitlinear_layer, tensorts_wrong_types):
    with pytest.raises(TypeError):
        bitlinear_layer._compute_mean_of_weights(tensorts_wrong_types)


def test__get_sign_of_weights_with_wrong_type(bitlinear_layer, tensorts_wrong_types):
    with pytest.raises(TypeError):
        bitlinear_layer._get_sign_of_weights(tensorts_wrong_types)


def test__get_sign_of_weights(bitlinear_layer):
    w = torch.tensor([[0.2, -0.1], [4.0, -2.1]])
    sign_w = bitlinear_layer._get_sign_of_weights(w)
    assert torch.equal(sign_w, torch.tensor([[1.0, -1.0], [1.0, -1.0]]))


def test__get_sign_has_expected_storage_size(bitlinear_layer):
    w = torch.tensor([[0.2, -0.1], [4.0, -2.1]])
    sign_w = bitlinear_layer._get_sign_of_weights(w)
    storage_size = sign_w.element_size() * sign_w.nelement()
    assert storage_size <= 4  # in bytes


# test quantization


def test_clip_activation_max_value():
    a = -1.0
    b = 1.0
    BATCH_SIZE = 2
    CONTEXT_LENGTH = 10
    MODEL_DIM = 12
    x = (
        torch.rand(
            BATCH_SIZE,
            CONTEXT_LENGTH,
            MODEL_DIM,
        )
        * 3.0
    )
    x_clipped = clip_activation(x, a, b)

    assert torch.max(x_clipped) <= b


def test_clip_activation_min_value():
    a = -1.0
    b = 1.0
    BATCH_SIZE = 2
    CONTEXT_LENGTH = 10
    MODEL_DIM = 12
    x = (
        torch.rand(
            BATCH_SIZE,
            CONTEXT_LENGTH,
            MODEL_DIM,
        )
        * -3.0
    )
    x_clipped = clip_activation(x, a, b)

    assert torch.min(x_clipped) >= a


def test_clip_activation():
    # arrange
    a = -2.0
    b = 2.0
    x = torch.tensor(
        [
            [
                [12.3, 1.8, 0.4, -90.6],
                [12.3, 1.8, 0.4, -90.6],
                [12.3, 1.8, 0.4, -90.6],
            ],
            [
                [12.3, 1.8, 0.4, -90.6],
                [12.3, 1.8, 0.4, -90.6],
                [12.3, 1.8, 0.4, -90.6],
            ],
        ]
    )
    # act
    x_clipped = clip_activation(x, a, b)

    # assert
    assert torch.equal(
        x_clipped,
        torch.tensor(
            [
                [
                    [2.0, 1.8, 0.4, -2.0],
                    [2.0, 1.8, 0.4, -2.0],
                    [2.0, 1.8, 0.4, -2.0],
                ],
                [
                    [2.0, 1.8, 0.4, -2.0],
                    [2.0, 1.8, 0.4, -2.0],
                    [2.0, 1.8, 0.4, -2.0],
                ],
            ]
        ),
    )


def test_compute_infinity_norm():
    x = torch.tensor([[[1.2, -2.3], [3.4, 1.2]],[[-6.7, 2.3], [-3.4, 1.2]]])

    gamma = compute_infinity_norm(x)

    assert torch.equal(gamma, torch.tensor([3.4, 6.7]))


@pytest.fixture(params=[
    torch.tensor([]),
        torch.tensor(2.0),
        torch.tensor([1.0]),
])
def activation_wrong_dims(request) -> List[torch.Tensor]:
    return request.param

def test_compute_infinity_norm_wrong_dim(activation_wrong_dims):
    with pytest.raises(DimensionError):
        compute_infinity_norm(activation_wrong_dims)

def test__compute_quantization_range():
    bit_precision = 8

    q_range = _compute_quantization_range(bit_precision)

    assert torch.equal(q_range, torch.tensor([-128, 128]))
