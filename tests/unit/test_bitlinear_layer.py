import numpy as np
import pytest
import torch
from bitllm.model import BitLinear
from bitllm.model import DimensionError

BATCH_SIZE = 2
CONTEXT_LENGTH = 5
MODEL_DIM = 3

x = torch.randn(BATCH_SIZE, CONTEXT_LENGTH, MODEL_DIM)
W = torch.tensor(
    [
        [-0.4235, -0.4627, -0.5140, -0.3553, -0.3684, 0.8928],
        [-0.5650, -0.6207, 0.0462, 0.4840, 0.3655, -1.4766],
        [-0.6663, -0.0252, 0.6200, -0.7174, 0.6964, 0.3969],
        [1.8214, 0.2993, -1.5880, 0.5250, 0.9086, 0.5998],
    ]
)

W0 = torch.zeros_like(W)

bitlinear = BitLinear()


@pytest.mark.parametrize("w", [W, W0])
def test__compute_mean_of_weights_n_x_m(w):
    n, m = w.size()
    average = w.sum() / (n * m)
    assert bitlinear._compute_mean_of_weights(w) == average


@pytest.mark.parametrize("w", [[1, 2, 3], np.array([[1.0, 2.0], [3.0, 4.0]]), "abc"])
def test__compute_mean_of_weights_with_bad_type(w):
    with pytest.raises(TypeError):
        bitlinear._compute_mean_of_weights(w)


@pytest.mark.parametrize(
    "w",
    [
        torch.tensor([]),
        torch.tensor(2.0),
        torch.tensor([1.0]),
        torch.tensor([[[1.0, 2.0], [1.0, 2.0]], [[1.0, 2.0], [1.0, 2.0]]]),
    ],
)
def test__compute_mean_of_weights_wrong_dim(w):
    with pytest.raises(DimensionError):
        bitlinear._compute_mean_of_weights(w)


def test__demean_weights_has_0_mean():
    demeaned_W = bitlinear._demean_weights(W)
    assert demeaned_W.mean().item() == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize("w", [[1, 2, 3], np.array([[1.0, 2.0], [3.0, 4.0]]), "abc"])
def test__demean_weights_with_bad_type(w):
    with pytest.raises(TypeError):
        bitlinear._demean_weights(w)


@pytest.mark.parametrize(
    "w",
    [
        torch.tensor([]),
        torch.tensor(2.0),
        torch.tensor([1.0]),
        torch.tensor([[[1.0, 2.0], [1.0, 2.0]], [[1.0, 2.0], [1.0, 2.0]]]),
    ],
)
def test__demean_weights_wrong_dim(w):
    with pytest.raises(DimensionError):
        bitlinear._demean_weights(w)


def test__get_sign_of_weights():
    w = torch.tensor([[0.2, -0.1], [4.0, -2.1]])
    sign_w = bitlinear._get_sign_of_weights(w)
    assert torch.equal(sign_w, torch.tensor([[1.0, -1.0], [1.0, -1.0]]))


def test__get_sign_has_expected_storage_size():
    w = torch.tensor([[0.2, -0.1], [4.0, -2.1]])
    sign_w = bitlinear._get_sign_of_weights(w)
    storage_size = sign_w.element_size() * sign_w.nelement()
    assert storage_size <= 4  # in bytes


@pytest.mark.parametrize("w", [[1, 2, 3], np.array([[1.0, 2.0], [3.0, 4.0]]), "abc"])
def test__get_sign_of_weights_with_bad_type(w):
    with pytest.raises(TypeError):
        bitlinear._get_sign_of_weights(w)


@pytest.mark.parametrize(
    "w",
    [
        torch.tensor([]),
        torch.tensor(2.0),
        torch.tensor([1.0]),
        torch.tensor([[[1.0, 2.0], [1.0, 2.0]], [[1.0, 2.0], [1.0, 2.0]]]),
    ],
)
def test__get_sign_of_weights_wrong_dim(w):
    with pytest.raises(DimensionError):
        bitlinear._get_sign_of_weights(w)
