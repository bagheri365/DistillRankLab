import numpy as np
import pytest
from distillranklab.comparison import Dataset, evaluate, make_dataset, softmax, train_student


def test_softmax_normalizes():
    np.testing.assert_allclose(softmax(np.array([[1000., 1001.]] )).sum(axis=-1), [1.])


def test_softmax_temperature_validation():
    with pytest.raises(ValueError):
        softmax(np.array([1., 2.]), 0)


def test_dataset_reproducible():
    a, b = make_dataset(10, seed=3), make_dataset(10, seed=3)
    np.testing.assert_array_equal(a.features, b.features)
    np.testing.assert_array_equal(a.teacher_logits, b.teacher_logits)


def test_data_shapes():
    d = make_dataset(12, seed=1)
    assert d.features.shape == (12, 4, 2)
    assert d.labels.shape == (12,)


@pytest.mark.parametrize('mode', ['hard', 'soft', 'hybrid'])
def test_training_improves_from_zero_weights(mode):
    d = make_dataset(300, seed=4, teacher_error_rate=0.1)
    baseline = evaluate(d, np.zeros(2))['accuracy']
    trained = evaluate(d, train_student(d, mode=mode))['accuracy']
    assert trained > baseline + 0.2


def test_holdout_evaluation():
    train = make_dataset(200, seed=2)
    test = make_dataset(200, seed=3)
    scores = evaluate(test, train_student(train, mode='hard'))
    assert scores['accuracy'] > 0.5
    assert 0 <= scores['teacher_agreement'] <= 1


def test_zero_steps_preserves_zero_weights():
    d = make_dataset(4, seed=1)
    np.testing.assert_array_equal(train_student(d, mode='hard', steps=0), [0., 0.])


def test_invalid_mode():
    with pytest.raises(ValueError):
        train_student(make_dataset(3, seed=1), mode='unknown')


def test_invalid_alpha():
    with pytest.raises(ValueError):
        train_student(make_dataset(3, seed=1), mode='hybrid', alpha=1.5)


def test_invalid_labels():
    d = make_dataset(3, seed=1)
    bad = Dataset(d.features, np.array([0, 1, 99]), d.teacher_logits)
    with pytest.raises(ValueError):
        train_student(bad, mode='hard')


def test_teacher_corruption_is_controlled():
    d = make_dataset(20, seed=5, teacher_error_rate=1.0)
    assert np.all(d.teacher_logits.argmax(axis=1) != d.labels)
