"""9B diagnostics: deterministic CPU-only fixtures."""
import numpy as np
import pytest

from distillranklab.learning_diagnostics import (learning_diagnostics, ordering_diagnostics,
                                                 teacher_agreement)
from distillranklab.real_data import Candidate, QueryGroup
from distillranklab.semantic_ablation import masked_features
from distillranklab.text_student import fit_text_head


def fixture():
    groups = [QueryGroup(f'q{i}', 'query', (Candidate(f'a{i}', 'positive', 1),
                                             Candidate(f'b{i}', 'negative', None),
                                             Candidate(f'c{i}', 'other', 0))) for i in range(4)]
    matrices = []
    for i in range(4):
        x = np.zeros((3, 10))
        x[:, 8] = [.9, .3, .2]
        x[:, 9] = [.4, .2, .1]
        x[:, 0] = [1, 0, .5]
        matrices.append(x)
    teacher = {(g.query_id, c.passage_id): float(3-j) for g in groups
               for j, c in enumerate(g.candidates)}
    return groups, matrices, teacher


def test_identical_order():
    g, _, t = fixture()
    result = ordering_diagnostics(g, t, t)
    assert result['changed_top1'] == 0
    assert result['changed_top10_order'] == 0
    assert result['mean_top10_membership_replacements'] == 0


def test_changed_order():
    g, _, t = fixture()
    reversed_scores = {k: -v for k, v in t.items()}
    result = ordering_diagnostics(g, t, reversed_scores)
    assert result['changed_top1'] == 4
    assert result['changed_full_order'] == 4


def test_teacher_perfect_agreement():
    g, _, t = fixture()
    a = teacher_agreement(g, t, t)
    assert a['teacher_top1_agreement'] == 1
    assert a['mean_spearman_rho'] == pytest.approx(1)


def test_teacher_reversed_agreement():
    g, _, t = fixture()
    a = teacher_agreement(g, t, {k: -v for k, v in t.items()})
    assert a['teacher_top1_agreement'] == 0
    assert a['mean_spearman_rho'] == pytest.approx(-1)


def test_trace_matches_uninstrumented_weights():
    g, x, t = fixture()
    trace = []
    a = fit_text_head(g, masked_features(x, 'combined'), t, steps=10, trace=trace)
    b = fit_text_head(g, masked_features(x, 'combined'), t, steps=10)
    assert a == b
    assert [p['step'] for p in trace] == [0, 1, 10]
    assert trace[-1]['weights'] == a['weights']
    assert trace[0]['soft_kl'] >= 0
    assert trace[-1]['soft_kl'] >= 0


def test_trace_teacher_only_reduces_kl():
    g, x, t = fixture()
    trace = []
    fit_text_head(g, masked_features(x, 'combined'), t, steps=20,
                  hard_weight=0, trace=trace)
    assert trace[-1]['soft_kl'] < trace[0]['soft_kl']


def test_diagnostics_reproducible():
    g, x, t = fixture()
    a = learning_diagnostics(g[:3], g[3:], x[:3], x[3:], t, steps=3)
    b = learning_diagnostics(g[:3], g[3:], x[:3], x[3:], t, steps=3)
    assert a == b
    assert set(a) == {'semantic_only', 'combined'}
    assert set(a['combined']) == {'teacher_only', 'hybrid', 'relevance_only'}
    assert a['combined']['hybrid']['trajectory'][-1]['step'] == 3


def test_bad_empty_groups():
    with pytest.raises(ValueError):
        ordering_diagnostics([], {}, {})
    with pytest.raises(ValueError):
        teacher_agreement([], {}, {})
    with pytest.raises(ValueError):
        learning_diagnostics([], [], [], [], {})


def test_bad_nonfinite_score():
    g, _, t = fixture()
    bad = dict(t)
    bad['q0', 'a0'] = float('nan')
    with pytest.raises(ValueError):
        ordering_diagnostics(g, t, bad)
    with pytest.raises(ValueError):
        teacher_agreement(g, bad, t)
