"""Nonlinear student reproducibility, shape validation, and objective checks."""
import json
import numpy as np
import pytest

from distillranklab.nonlinear_student import fit_nonlinear, predict_nonlinear, parameter_count
from distillranklab.real_data import Candidate, QueryGroup
from distillranklab.real_student import save_student


def groups():
    return tuple(QueryGroup(str(i), 'alpha beta', (
        Candidate('a', 'alpha beta evidence', 1),
        Candidate('b', 'gamma unrelated', None),
        Candidate('c', 'beta text', 0))) for i in range(6))


def teacher():
    return {(g.query_id, c.passage_id): float(3 - j) for g in groups()
            for j, c in enumerate(g.candidates)}


@pytest.mark.parametrize('mode', ['hard', 'soft', 'hybrid'])
def test_fit_predict_modes(mode):
    model = fit_nonlinear(groups(), None if mode == 'hard' else teacher(), mode=mode, steps=4)
    scores = predict_nonlinear(groups(), model)
    assert len(scores) == 18 and np.isfinite(list(scores.values())).all()
    assert model['parameter_count'] == 81


def test_determinism_and_seed():
    a = fit_nonlinear(groups(), teacher(), steps=5)
    b = fit_nonlinear(groups(), teacher(), steps=5)
    assert a == b
    c = fit_nonlinear(groups(), teacher(), steps=5, seed=43)
    assert a['w1'] != c['w1']


def test_save_reload(tmp_path):
    model = fit_nonlinear(groups(), teacher(), steps=2)
    path = tmp_path / 'model.json'
    save_student(model, path)
    assert predict_nonlinear(groups(), json.loads(path.read_text())) == predict_nonlinear(groups(), model)


def test_label_free_inference():
    model = fit_nonlinear(groups(), teacher(), steps=2)
    unlabeled = tuple(QueryGroup(g.query_id, g.query, tuple(
        Candidate(c.passage_id, c.passage, None) for c in g.candidates)) for g in groups())
    assert predict_nonlinear(groups(), model) == predict_nonlinear(unlabeled, model)


@pytest.mark.parametrize('kwargs', [
    {'hidden_units': 0}, {'hidden_units': True}, {'steps': 0},
    {'temperature': 0}, {'learning_rate': -1}, {'weight_decay': -1},
    {'hard_weight': 2}, {'feature_set': 'unknown'}, {'mode': 'wrong'},
])
def test_bad_settings(kwargs):
    with pytest.raises(ValueError):
        fit_nonlinear(groups(), teacher(), **({'steps': 2} | kwargs))


def test_missing_teacher():
    with pytest.raises(ValueError, match='teacher scores'):
        fit_nonlinear(groups(), None, mode='soft', steps=2)


def test_missing_hard_positives():
    empty = tuple(QueryGroup(g.query_id, g.query, tuple(
        Candidate(c.passage_id, c.passage, None) for c in g.candidates)) for g in groups())
    with pytest.raises(ValueError, match='retrieved positives'):
        fit_nonlinear(empty, None, mode='hard', steps=2)


def test_corrupt_model():
    model = fit_nonlinear(groups(), teacher(), steps=2)
    model['w2'] = [float('nan')] * 8
    with pytest.raises(ValueError, match='invalid nonlinear'):
        predict_nonlinear(groups(), model)


def test_count():
    assert parameter_count(8, 8) == 81
    assert parameter_count(8, 16) == 161
    with pytest.raises(ValueError):
        parameter_count(0, 8)
