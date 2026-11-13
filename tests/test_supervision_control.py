"""9A supervision controls: deterministic, CPU-only, no downloads."""
import numpy as np
import pytest

from distillranklab.real_data import Candidate, QueryGroup
from distillranklab.supervision_control import SUPERVISION, cosine_baseline, supervision_ablation
from distillranklab.text_student import fit_text_head


def fixtures():
    groups = [QueryGroup(query_id=f'q{i}', query=f'query {i}', candidates=(
        Candidate(passage_id=f'd{i}a', passage='relevant text', relevance=1),
        Candidate(passage_id=f'd{i}b', passage='other text', relevance=None))) for i in range(3)]
    matrices = []
    for i in range(3):
        x = np.zeros((2, 10))
        x[:, 8] = [.8 - .1 * i, .2 + .1 * i]
        x[:, 9] = [.1, .3]
        x[:, 0] = [1, 0]
        matrices.append(x)
    teacher = {(g.query_id, c.passage_id): float(2 - j)
               for g in groups for j, c in enumerate(g.candidates)}
    return groups, matrices, teacher


def test_weights():
    assert SUPERVISION == {'teacher_only': 0., 'hybrid': .5, 'relevance_only': 1.}


def test_cosine_uses_only_coordinate_eight():
    groups, matrices, _ = fixtures()
    scores = cosine_baseline(groups, matrices)
    assert scores['q0', 'd0a'] == pytest.approx(.8)
    assert scores['q0', 'd0b'] == pytest.approx(.2)


def test_cosine_bad_length():
    groups, matrices, _ = fixtures()
    with pytest.raises(ValueError):
        cosine_baseline(groups, matrices[:1])


def test_cosine_bad_matrix():
    groups, matrices, _ = fixtures()
    matrices[0][0, 8] = np.nan
    with pytest.raises(ValueError):
        cosine_baseline(groups, matrices)


def test_cosine_duplicate_keys():
    groups, matrices, _ = fixtures()
    with pytest.raises(ValueError):
        cosine_baseline(groups + groups[:1], matrices + matrices[:1])


@pytest.mark.parametrize('block,expected', [('semantic_only', 4), ('combined', 3)])
def test_ablation_modes(block, expected):
    groups, matrices, teacher = fixtures()
    report, scores = supervision_ablation(groups[:2], groups[2:], matrices[:2], matrices[2:],
                                          teacher, block=block, steps=4)
    assert len(report) == len(scores) == expected
    for name in ('teacher_only', 'hybrid', 'relevance_only'):
        assert len(scores[name]) == 2
        assert report[name]['metrics']['total_queries'] == 1
    if block == 'semantic_only':
        assert report['untrained_cosine']['effective_head_parameters'] == 0


def test_determinism():
    groups, matrices, teacher = fixtures()
    a, scores_a = supervision_ablation(groups[:2], groups[2:], matrices[:2], matrices[2:], teacher, steps=3)
    b, scores_b = supervision_ablation(groups[:2], groups[2:], matrices[:2], matrices[2:], teacher, steps=3)
    assert a == b and scores_a == scores_b


def test_invalid_block():
    groups, matrices, teacher = fixtures()
    with pytest.raises(ValueError):
        supervision_ablation(groups, groups, matrices, matrices, teacher, block='lexical_only')


def test_missing_teacher_fails_closed():
    groups, matrices, _ = fixtures()
    with pytest.raises(KeyError):
        supervision_ablation(groups, groups, matrices, matrices, {}, steps=1)


def test_hard_only_is_independent_of_teacher_values():
    groups, matrices, teacher = fixtures()
    changed = {k: -v * 100 for k, v in teacher.items()}
    # Teacher-only/hybrid differ, but relevance-only must be identical.
    _, a = supervision_ablation(groups[:2], groups[2:], matrices[:2], matrices[2:], teacher, steps=5)
    _, b = supervision_ablation(groups[:2], groups[2:], matrices[:2], matrices[2:], changed, steps=5)
    assert a['relevance_only'] == b['relevance_only']


def test_teacher_only_ignores_hard_labels():
    groups, matrices, teacher = fixtures()
    unlabeled = [QueryGroup(query_id=g.query_id, query=g.query, candidates=tuple(
        Candidate(passage_id=c.passage_id, passage=c.passage, relevance=None)
        for c in g.candidates)) for g in groups]
    a = fit_text_head(groups[:2], matrices[:2], teacher, steps=4, hard_weight=0.)
    b = fit_text_head(unlabeled[:2], matrices[:2], teacher, steps=4, hard_weight=0.)
    assert a == b
