import json
import numpy as np
import pytest
from distillranklab.real_data import Candidate, QueryGroup
from distillranklab.real_student import (FEATURE_NAMES, feature_groups, fit_student,
                                         predict_student, save_student, evaluate_student)


def groups():
    return (QueryGroup('q1', 'alpha beta', (
        Candidate('a', 'alpha beta beta', 1), Candidate('b', 'gamma delta', None),
        Candidate('c', 'alpha', 0))),
        QueryGroup('q2', 'gamma', (
            Candidate('d', 'delta', None), Candidate('e', 'gamma gamma', 2))))


def teacher():
    return {('q1', 'a'): 4., ('q1', 'b'): -2., ('q1', 'c'): 1.,
            ('q2', 'd'): -1., ('q2', 'e'): 3.}


def test_features_are_finite_and_label_free():
    g = groups()
    x = feature_groups(g)
    assert [v.shape for v in x] == [(3, len(FEATURE_NAMES)), (2, len(FEATURE_NAMES))]
    assert np.isfinite(x[0]).all()
    altered = (QueryGroup('q1', 'alpha beta', tuple(Candidate(c.passage_id, c.passage, None) for c in g[0].candidates)), g[1])
    np.testing.assert_array_equal(x[0], feature_groups(altered)[0])


@pytest.mark.parametrize('mode', ['soft', 'hard', 'hybrid'])
def test_fit_and_predict(mode):
    m = fit_student(groups(), teacher() if mode != 'hard' else None, mode=mode, steps=60)
    scores = predict_student(groups(), m)
    assert len(scores) == 5 and all(np.isfinite(list(scores.values())))
    assert evaluate_student(groups(), m)['total_queries'] == 2


def test_deterministic():
    a = fit_student(groups(), teacher(), mode='hybrid', steps=20)
    b = fit_student(groups(), teacher(), mode='hybrid', steps=20)
    assert a == b


def test_hard_requires_positive():
    g = (QueryGroup('q', 'alpha', (Candidate('x', 'alpha', None),)),)
    with pytest.raises(ValueError, match='positive'):
        fit_student(g, None, mode='hard')


def test_soft_can_use_no_judgments():
    g = (QueryGroup('q', 'alpha', (Candidate('x', 'alpha', None), Candidate('y', 'beta', None))),)
    m = fit_student(g, {('q', 'x'): 1., ('q', 'y'): 0.}, steps=3)
    assert m['training_queries'] == 1


def test_missing_teacher_score_rejected():
    with pytest.raises(KeyError):
        fit_student(groups(), {}, mode='soft')


@pytest.mark.parametrize('kwargs', [dict(mode='bad'), dict(temperature=0),
                                    dict(hard_weight=-1), dict(steps=0),
                                    dict(learning_rate=0)])
def test_bad_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        fit_student(groups(), teacher(), **kwargs)


def test_model_roundtrip(tmp_path):
    m = fit_student(groups(), teacher(), steps=4)
    p = tmp_path / 'model.json'
    save_student(m, p)
    assert predict_student(groups(), json.loads(p.read_text())) == predict_student(groups(), m)


def test_invalid_model_rejected():
    m = fit_student(groups(), teacher(), steps=2)
    m['features'] = ['wrong']
    with pytest.raises(ValueError):
        predict_student(groups(), m)
