import math

import pytest

from distillranklab.query_errors import query_error_report
from distillranklab.real_data import Candidate, QueryGroup


def fixture():
    groups = (
        QueryGroup('q1', 'alpha beta', (Candidate('a', 'alpha beta', 1), Candidate('b', 'z', None))),
        QueryGroup('q2', 'alpha', (Candidate('c', 'z', None), Candidate('d', 'alpha', 1))),
        QueryGroup('q3', 'missing positive', (Candidate('e', 'x', None), Candidate('f', 'y', None))),
    )
    bm25 = {(g.query_id, c.passage_id): -float(i) for g in groups for i, c in enumerate(g.candidates)}
    semantic = dict(bm25)
    semantic[('q1', 'a')], semantic[('q1', 'b')] = -1., 0.
    semantic[('q2', 'c')], semantic[('q2', 'd')] = -1., 0.
    combined = dict(bm25)
    return groups, {'bm25': bm25, 'semantic_only': semantic, 'combined': combined}


def test_counts_and_delta_consistency():
    groups, scores = fixture()
    report = query_error_report(groups, scores)
    assert report['official_test_used'] is False
    for item in report['comparisons'].values():
        assert item['queries'] == 3
        assert (item['improved'], item['regressed'], item['unchanged']) == (1, 1, 1)
        assert math.isclose(item['mean_delta_ndcg'], 0., abs_tol=1e-12)
        assert sum(v['queries'] for v in item['by_positive_bm25_rank'].values()) == 3
        assert sum(v['queries'] for v in item['by_query_length'].values()) == 3
        assert sum(v['queries'] for v in item['by_top1_overlap'].values()) == 3
        assert item['top_gains'][0]['query_id'] == 'q2'
        assert item['top_losses'][0]['query_id'] == 'q1'
        assert len(item['per_query']) == 3


def test_positive_rank_and_missing_judgments():
    groups, scores = fixture()
    rows = query_error_report(groups, scores)['comparisons']['semantic_only_vs_bm25']['per_query']
    by_id = {r['query_id']: r for r in rows}
    assert by_id['q1']['best_relevant_bm25_rank'] == 1
    assert by_id['q2']['best_relevant_bm25_rank'] == 2
    assert by_id['q3']['best_relevant_bm25_rank'] is None
    assert by_id['q3']['positive_rank_bin'] == 'not_retrieved'
    assert by_id['q3']['unjudged_candidates'] == 2


def test_example_limit_zero():
    groups, scores = fixture()
    result = query_error_report(groups, scores, examples=0)
    assert not result['comparisons']['semantic_only_vs_bm25']['top_gains']


@pytest.mark.parametrize('count', [-1, 51, True, 1.2])
def test_invalid_examples(count):
    groups, scores = fixture()
    with pytest.raises(ValueError, match='examples'):
        query_error_report(groups, scores, examples=count)


def test_missing_model_rejected():
    groups, scores = fixture()
    del scores['combined']
    with pytest.raises(ValueError, match='required rankers'):
        query_error_report(groups, scores)


def test_missing_score_rejected():
    groups, scores = fixture()
    del scores['semantic_only'][('q1', 'a')]
    with pytest.raises(ValueError, match='pool mismatch'):
        query_error_report(groups, scores)


def test_reordering_groups_does_not_change_results():
    groups, scores = fixture()
    a = query_error_report(groups, scores)
    b = query_error_report(groups[::-1], scores)
    assert a == b
