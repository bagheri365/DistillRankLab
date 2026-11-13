"""10B robustness checks, independent of labels for ranking decisions."""
import numpy as np
import pytest

from distillranklab.real_data import Candidate, QueryGroup
from distillranklab.reranking_robustness import (
    SHORTLIST, audit_pool, paired_audit, positive_rank_transitions,
    shortlist_audit, shortlist_scores, top_ids,
)


def fixture():
    groups, matrices = [], []
    for q, relevant in [('a', 0), ('b', 3), ('c', None)]:
        groups.append(QueryGroup(q, 'query', tuple(
            Candidate(str(i), f'doc{i}', 1 if i == relevant else None) for i in range(5))))
        x = np.zeros((5, 10))
        x[:, 8] = [.1, .2, .3, .4, .5]
        matrices.append(x)
    return tuple(groups), tuple(matrices)


def test_shortlist_frozen():
    groups, matrices = fixture()
    assert tuple(shortlist_scores(groups, matrices)) == SHORTLIST
    assert len(shortlist_scores(groups, matrices)['blend_0.75']) == 15


def test_label_independent_shortlist():
    groups, matrices = fixture()
    unlabeled = tuple(QueryGroup(g.query_id, g.query, tuple(
        Candidate(c.passage_id, c.passage, None) for c in g.candidates)) for g in groups)
    assert shortlist_scores(groups, matrices) == shortlist_scores(unlabeled, matrices)


def test_top_ids_ties_preserve_bm25_order():
    groups, _ = fixture()
    scores = {(g.query_id, c.passage_id): 1.0 for g in groups for c in g.candidates}
    assert top_ids(groups, scores, k=2)['a'] == ('0', '1')
    scores['a', '0'] = float('nan')
    with pytest.raises(ValueError):
        top_ids(groups, scores)
    with pytest.raises(ValueError):
        top_ids(groups, scores, k=0)


def test_positive_transitions_and_unjudged():
    groups, matrices = fixture()
    scores = shortlist_scores(groups, matrices)
    result = positive_rank_transitions(groups, scores['bm25'], scores['cosine'])
    assert sum(result['counts'].values()) == 3
    assert result['counts']['not_retrieved -> not_retrieved'] == 1
    rows = {r['query_id']: r for r in result['queries']}
    assert rows['a']['bm25_positive_rank'] == 1
    assert rows['a']['challenger_positive_rank'] == 5
    assert rows['b']['bm25_positive_rank'] == 4
    assert rows['b']['challenger_positive_rank'] == 2


def test_paired_audit_leave_one_out_and_no_positive():
    groups, matrices = fixture()
    scores = shortlist_scores(groups, matrices)
    result = paired_audit(groups, scores['bm25'], scores['cosine'], top_n=2)
    assert result['query_count'] == 3
    assert result['improved'] + result['regressed'] + result['unchanged'] == 3
    assert len(result['top_gains']) == len(result['top_losses']) == 2
    assert len(result['per_query']) == 3
    assert result['leave_one_out_mean_delta_min'] <= result['leave_one_out_mean_delta_max']
    with pytest.raises(ValueError):
        paired_audit(groups, scores['bm25'], scores['cosine'], top_n=0)


def test_pool_truncation_retrieval_coverage_and_consistency():
    groups, matrices = fixture()
    report = audit_pool(groups, matrices, cutoffs=(2, 5))
    assert report['2']['candidate_count'] == 6
    assert report['5']['candidate_count'] == 15
    assert report['2']['models']['bm25']['queries_with_retrieved_positive'] == 1
    assert report['5']['models']['bm25']['queries_with_retrieved_positive'] == 2
    assert report['5']['models']['bm25']['ndcg_at_k_all_queries'] == pytest.approx(
        shortlist_audit(groups, matrices)['models']['bm25']['ndcg_at_k_all_queries'])
    with pytest.raises(ValueError):
        audit_pool(groups, matrices, cutoffs=(1,))
    with pytest.raises(ValueError):
        audit_pool(groups, matrices, cutoffs=(2, 2))


def test_shortlist_audit_reports_all_baselines():
    groups, matrices = fixture()
    audit = shortlist_audit(groups, matrices)
    assert set(audit['models']) == set(SHORTLIST)
    assert set(audit['comparisons_vs_bm25']) == set(SHORTLIST) - {'bm25'}
    assert set(audit['comparisons_vs_cosine']) == {'blend_0.5', 'blend_0.75'}
    assert all(0 <= n <= 3 for n in audit['top1_agreement'].values())
