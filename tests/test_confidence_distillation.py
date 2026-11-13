"""Confidence-weighted training is deterministic and label-free for weights."""
import numpy as np
import pytest
from distillranklab.real_data import Candidate, QueryGroup
from distillranklab.confidence_distillation import confidence_weights, fit_weighted_head, confidence_ablation
from distillranklab.text_student import fit_text_head


def sample():
    groups, xs, teacher = [], [], {}
    for i in range(5):
        g = QueryGroup(query_id=str(i), query='q', candidates=(
            Candidate(passage_id='a', passage='a', relevance=1),
            Candidate(passage_id='b', passage='b', relevance=None),
            Candidate(passage_id='c', passage='c', relevance=None)))
        groups.append(g)
        x = np.zeros((3, 10)); x[:, 0] = [1, .5, .1]; x[:, 8] = [.8, .5, .1]
        xs.append(x)
        for j, c in enumerate(g.candidates):
            teacher[g.query_id, c.passage_id] = float([i + 1, 2, 0][j])
    return groups, xs, teacher


def test_uniform_reproduces_original_hybrid():
    g, x, t = sample()
    a = fit_weighted_head(g, x, t, np.ones(len(g)), steps=7)
    b = fit_text_head(g, x, t, steps=7)
    np.testing.assert_allclose(a['weights'], b['weights'], atol=0, rtol=0)


@pytest.mark.parametrize('scheme', ['uniform', 'margin', 'agreement', 'shuffled_margin'])
def test_weights_positive_normalized_deterministic(scheme):
    g, _, t = sample()
    a = confidence_weights(g, t, scheme=scheme)
    assert np.all(a > 0) and a.mean() == pytest.approx(1)
    np.testing.assert_array_equal(a, confidence_weights(g, t, scheme=scheme))


def test_weight_rules_do_not_use_labels():
    g, _, t = sample()
    unlabeled = [QueryGroup(query_id=v.query_id, query=v.query, candidates=tuple(
        Candidate(passage_id=c.passage_id, passage=c.passage, relevance=None)
        for c in v.candidates)) for v in g]
    for scheme in ('uniform', 'margin', 'agreement', 'shuffled_margin'):
        np.testing.assert_array_equal(confidence_weights(g, t, scheme=scheme),
                                      confidence_weights(unlabeled, t, scheme=scheme))


def test_shuffle_preserves_distribution():
    g, _, t = sample()
    np.testing.assert_allclose(sorted(confidence_weights(g, t, scheme='margin')),
                               sorted(confidence_weights(g, t, scheme='shuffled_margin')))


def test_ablation_returns_four_heads():
    g, x, t = sample()
    metrics, predictions = confidence_ablation(g[:4], g[4:], x[:4], x[4:], t, steps=3)
    assert set(metrics) == set(predictions) == {'uniform', 'margin', 'agreement', 'shuffled_margin'}
    assert all(len(p) == 3 for p in predictions.values())


def test_invalid_weights_fail():
    g, x, t = sample()
    with pytest.raises(ValueError):
        fit_weighted_head(g, x, t, [1, 1, 1, 1, -1], steps=1)
    with pytest.raises(ValueError):
        fit_weighted_head(g, x, t, [2] * 5, steps=1)


def test_missing_teacher_fails():
    g, _, t = sample()
    del t['0', 'a']
    with pytest.raises(KeyError):
        confidence_weights(g, t, scheme='margin')


def test_unknown_scheme_fails():
    g, _, t = sample()
    with pytest.raises(ValueError):
        confidence_weights(g, t, scheme='oracle')
