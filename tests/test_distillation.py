import numpy as np
import pytest

from distillranklab.distillation import (
    distillation_loss,
    kl_divergence,
    softmax,
)


def test_softmax_sums_to_one():
    assert softmax([1, 2, 3]).sum() == pytest.approx(1.0)


def test_softmax_preserves_order():
    assert np.argmax(softmax([1, 3, 2])) == 1


def test_softmax_shift_invariance():
    np.testing.assert_allclose(softmax([1, 2, 3]), softmax([101, 102, 103]))


def test_softmax_handles_large_logits():
    assert np.isfinite(softmax([10000, 10001])).all()


def test_softmax_temperature_softens():
    assert softmax([3, 2, 0], temperature=4)[0] < softmax([3, 2, 0], temperature=1)[0]


@pytest.mark.parametrize("temperature", [0, -1, np.inf, np.nan])
def test_softmax_rejects_bad_temperature(temperature):
    with pytest.raises(ValueError):
        softmax([1, 2], temperature)


def test_softmax_batched():
    assert softmax([[1, 2], [2, 3]]).shape == (2, 2)


def test_kl_same_distribution_is_zero():
    assert kl_divergence([0.25, 0.75], [0.25, 0.75]) == pytest.approx(0.0)


def test_kl_known_value():
    assert kl_divergence([0.5, 0.5], [0.25, 0.75]) == pytest.approx(0.14384103622589045)


def test_kl_rejects_invalid_probabilities():
    with pytest.raises(ValueError):
        kl_divergence([0.5, 0.5], [0.5, 0.6])


def test_distillation_loss_zero_for_identical_logits():
    assert distillation_loss([3, 2, 0], [3, 2, 0]) == pytest.approx(0.0)


def test_distillation_loss_positive_for_different_logits():
    assert distillation_loss([3, 2, 0], [1, 2, 0]) > 0


def test_distillation_loss_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        distillation_loss([1, 2, 3], [1, 2])
