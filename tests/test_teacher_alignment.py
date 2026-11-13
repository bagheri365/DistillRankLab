"""9C deterministic, sparse-judgment-aware teacher audit fixtures."""
import pytest

from distillranklab.real_data import Candidate, QueryGroup
from distillranklab.teacher_alignment import teacher_relevance_alignment


def fixture():
    groups = (
        QueryGroup('q1', 'a', (Candidate('p1', 'positive', 1), Candidate('p2', 'unknown', None), Candidate('p3', 'zero', 0))),
        QueryGroup('q2', 'b', (Candidate('p4', 'unknown', None), Candidate('p5', 'positive', 1), Candidate('p6', 'zero', 0))),
        QueryGroup('q3', 'c', (Candidate('p7', 'unknown', None), Candidate('p8', 'zero', 0))),
    )
    teacher = {('q1', 'p1'): 1, ('q1', 'p2'): 3, ('q1', 'p3'): 0,
               ('q2', 'p4'): 0, ('q2', 'p5'): 3, ('q2', 'p6'): 1,
               ('q3', 'p7'): 1, ('q3', 'p8'): 2}
    return groups, teacher


def test_sparse_judgments_and_ranks():
    groups, scores = fixture()
    report = teacher_relevance_alignment(groups, scores)
    rows = {r['query_id']: r for r in report['per_query']}
    assert report['queries'] == 3
    assert rows['q1']['bm25_best_positive_rank'] == 1
    assert rows['q1']['teacher_best_positive_rank'] == 2
    assert rows['q1']['teacher_top1_judgment'] == 'unjudged'
    assert rows['q1']['positive_margin_vs_best_other_including_unjudged'] == -2
    assert rows['q1']['positive_margin_vs_best_judged_zero'] == 1
    assert rows['q2']['bm25_best_positive_rank'] == 2
    assert rows['q2']['teacher_best_positive_rank'] == 1
    assert rows['q3']['bm25_best_positive_rank'] is None
    assert rows['q3']['teacher_best_positive_rank'] is None
    assert rows['q3']['teacher_top1_judgment'] == 'judged_nonrelevant'
    assert report['teacher_bm25_top1_disagreement'] == 3
    assert report['positive_rank_summary']['teacher_better_positive_rank'] == 1
    assert report['positive_rank_summary']['teacher_worse_positive_rank'] == 1
    assert report['by_bm25_positive_rank']['no_retrieved_positive']['queries'] == 1
    assert report['teacher_top1_judgments']['unjudged'] == 1
    assert report['teacher_top1_judgments']['relevant'] == 1
    assert report['judged_zero_margin_available'] == 2


def test_stable_ties_and_determinism():
    groups, scores = fixture()
    scores['q1', 'p1'] = 3
    result = teacher_relevance_alignment(groups, scores)
    assert result['per_query'][0]['teacher_best_positive_rank'] == 1
    assert result == teacher_relevance_alignment(groups, scores)


def test_missing_and_extra_scores_rejected():
    groups, scores = fixture()
    with pytest.raises(ValueError, match='mismatch'):
        teacher_relevance_alignment(groups, dict(list(scores.items())[1:]))
    with pytest.raises(ValueError, match='mismatch'):
        teacher_relevance_alignment(groups, {**scores, ('other', 'x'): 1})


def test_nonfinite_scores_rejected():
    groups, scores = fixture()
    with pytest.raises(ValueError, match='nonfinite'):
        teacher_relevance_alignment(groups, {**scores, ('q1', 'p1'): float('nan')})


def test_empty_and_bad_arguments():
    groups, scores = fixture()
    with pytest.raises(ValueError):
        teacher_relevance_alignment([], {})
    with pytest.raises(ValueError):
        teacher_relevance_alignment(groups, scores, k=0)
    with pytest.raises(ValueError):
        teacher_relevance_alignment(groups, scores, examples=-1)


def test_no_judged_zero_margin_is_null():
    group = QueryGroup('q', 'q', (Candidate('a', 'a', 1), Candidate('b', 'b', None)))
    result = teacher_relevance_alignment([group], {('q', 'a'): 2., ('q', 'b'): 1.})
    row = result['per_query'][0]
    assert row['positive_margin_vs_best_judged_zero'] is None
    assert result['judged_zero_margin_available'] == 0


def test_top_gains_losses_bounded():
    groups, scores = fixture()
    result = teacher_relevance_alignment(groups, scores, examples=1)
    assert len(result['top_gains']) == 1
    assert len(result['top_losses']) == 1
