"""Feature-set integrity and train-only ablation compatibility."""
import json

import numpy as np
import pytest

from distillranklab.real_data import Candidate, QueryGroup
from distillranklab.real_student import FEATURE_SETS, feature_groups, fit_student, predict_student, save_student
from distillranklab.validation import select_student


def sample():
    return tuple(QueryGroup(str(i), 'alpha beta', (
        Candidate('a', 'alpha beta gamma', 1), Candidate('b', 'gamma delta', None),
        Candidate('c', 'beta alpha', 0))) for i in range(6))


def scores():
    return {(g.query_id, c.passage_id): float(3 - i) for g in sample()
            for i, c in enumerate(g.candidates)}


@pytest.mark.parametrize('feature_set', list(FEATURE_SETS))
def test_features_shape_finite_and_label_free(feature_set):
    groups = sample()
    xs = feature_groups(groups, feature_set=feature_set)
    assert all(x.shape == (3, len(FEATURE_SETS[feature_set])) for x in xs)
    assert all(np.isfinite(x).all() for x in xs)
    unlabeled = tuple(QueryGroup(g.query_id, g.query, tuple(
        Candidate(c.passage_id, c.passage, None) for c in g.candidates)) for g in groups)
    for x, y in zip(xs, feature_groups(unlabeled, feature_set=feature_set)):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize('feature_set', list(FEATURE_SETS))
@pytest.mark.parametrize('mode', ['hard', 'soft', 'hybrid'])
def test_fit_predict_roundtrip(feature_set, mode, tmp_path):
    model = fit_student(sample(), None if mode == 'hard' else scores(),
                        feature_set=feature_set, mode=mode, steps=3)
    assert model['features'] == list(FEATURE_SETS[feature_set])
    result = predict_student(sample(), model)
    assert len(result) == 18 and all(np.isfinite(list(result.values())))
    path = tmp_path / 'student.json'
    save_student(model, path)
    assert predict_student(sample(), json.loads(path.read_text())) == result


def test_legacy_model_without_feature_set():
    model = fit_student(sample(), scores(), steps=2)
    model.pop('feature_set')
    assert len(predict_student(sample(), model)) == 18


def test_bad_feature_set_and_model():
    with pytest.raises(ValueError, match='feature set'):
        feature_groups(sample(), feature_set='missing')
    with pytest.raises(ValueError, match='feature set'):
        fit_student(sample(), scores(), feature_set='missing')
    model = fit_student(sample(), scores(), feature_set='rich', steps=2)
    model['feature_set'] = 'baseline'
    with pytest.raises(ValueError, match='incompatible'):
        predict_student(sample(), model)


def test_query_disjoint_ablation_selection():
    configs = [dict(mode='hard', learning_rate=0.05, steps=2, temperature=2.0,
                    hard_weight=1.0, feature_set=feature_set) for feature_set in FEATURE_SETS]
    report, model = select_student(sample(), None, configs=configs)
    assert not set(report['fit_query_ids']) & set(report['validation_query_ids'])
    assert len(report['trials']) == 3
    assert model['feature_set'] in FEATURE_SETS


def test_bad_config_key_rejected():
    config = dict(mode='hard', learning_rate=0.05, steps=2, temperature=2.0,
                  hard_weight=1.0, feature_set='baseline', typo=True)
    with pytest.raises(ValueError, match='config keys'):
        select_student(sample(), None, configs=[config])
