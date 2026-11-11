import numpy as np
import pytest

from distillranklab.student import LinearStudent, loss_and_gradient, train_student


@pytest.fixture
def sample():
    x = np.array([[[1., 0.], [0., 1.], [2., 0.]],
                  [[0., 2.], [1., 0.], [1., 1.]]])
    return x, x @ np.array([1.5, -0.5])


def test_predict_shape(sample):
    x, _ = sample
    assert LinearStudent(np.array([1., 2.])).predict(x).shape == (2, 3)


def test_gradient_matches_finite_difference(sample):
    x, teacher = sample
    w = np.array([0.3, -0.2])
    _, gradient = loss_and_gradient(x, teacher, w, temperature=2)
    epsilon = 1e-6
    for i in range(len(w)):
        offset = np.zeros_like(w)
        offset[i] = epsilon
        plus, _ = loss_and_gradient(x, teacher, w + offset, temperature=2)
        minus, _ = loss_and_gradient(x, teacher, w - offset, temperature=2)
        assert gradient[i] == pytest.approx((plus - minus) / (2 * epsilon), abs=1e-6)


def test_training_reduces_loss_and_recovers_ranking(sample):
    x, teacher = sample
    student, history = train_student(x, teacher, steps=500, learning_rate=0.1)
    assert history[-1] < history[0] * 0.01
    np.testing.assert_array_equal(np.argmax(student.predict(x), axis=1),
                                  np.argmax(teacher, axis=1))


def test_training_is_reproducible(sample):
    x, teacher = sample
    a, h1 = train_student(x, teacher, steps=5, seed=17)
    b, h2 = train_student(x, teacher, steps=5, seed=17)
    np.testing.assert_array_equal(a.weights, b.weights)
    np.testing.assert_array_equal(h1, h2)


@pytest.mark.parametrize("bad", [0, -1])
def test_rejects_invalid_steps(sample, bad):
    with pytest.raises(ValueError, match="steps"):
        train_student(*sample, steps=bad)


def test_rejects_mismatched_teacher_shape(sample):
    x, _ = sample
    with pytest.raises(ValueError, match="teacher_logits"):
        train_student(x, np.zeros((2, 2)))


def test_rejects_invalid_learning_rate(sample):
    with pytest.raises(ValueError, match="learning_rate"):
        train_student(*sample, learning_rate=0)


def test_rejects_nonfinite_features(sample):
    x, teacher = sample
    x[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        train_student(x, teacher)
