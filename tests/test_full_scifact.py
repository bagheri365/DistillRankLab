"""Full-split controls and per-query benchmark diagnostics (offline)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from distillranklab.benchmark import (bm25_order_scores, compare_rankers,
                                      per_query_comparison, summarize_ranking)
from distillranklab.real_data import Candidate, QueryGroup


def sample():
    return (
        QueryGroup('q2', 'two', (Candidate('a', 'wrong', None), Candidate('b', 'right', 1))),
        QueryGroup('q1', 'one', (Candidate('c', 'right', 1), Candidate('d', 'wrong', None))),
        QueryGroup('q3', 'three', (Candidate('e', 'unknown', None),)),
    )


def test_per_query_improvement_regression_and_miss():
    groups = sample()
    baseline = bm25_order_scores(groups)
    teacher = dict(baseline)
    teacher['q2', 'b'] = 10  # improves
    teacher['q1', 'd'] = 10  # regresses
    report = per_query_comparison(groups, baseline, teacher)
    assert [r['query_id'] for r in report['queries']] == ['q1', 'q2', 'q3']
    assert (report['ndcg_improved_queries'], report['ndcg_regressed_queries'],
            report['ndcg_unchanged_queries']) == (1, 1, 1)
    assert report['queries'][2]['retrieved_positive'] is False
    assert report['queries'][2]['teacher_ndcg_at_k'] == 0
    assert report['queries'][2]['bm25_ndcg_at_k'] == 0


def test_per_query_averages_match_all_query_metrics():
    groups = sample()
    scores = bm25_order_scores(groups)
    report = per_query_comparison(groups, scores, scores, k=1)
    summary = summarize_ranking(groups, scores, k=1)
    assert sum(r['bm25_mrr_at_k'] for r in report['queries']) / 3 == pytest.approx(summary['mrr_at_k_all_queries'])
    assert sum(r['bm25_ndcg_at_k'] for r in report['queries']) / 3 == pytest.approx(summary['ndcg_at_k_all_queries'])


def test_per_query_judged_only():
    groups = sample()
    scores = bm25_order_scores(groups)
    report = per_query_comparison(groups, scores, scores, unjudged='judged_only')
    assert report['queries'][1]['bm25_mrr_at_k'] == 1


def test_comparison_opt_in_diagnostics_and_cache(tmp_path):
    groups = sample()
    result = compare_rankers(groups, scorer=lambda pairs: [1.0] * len(pairs),
                             model_id='mock', cache_path=tmp_path / 'cache.json', per_query=True)
    assert result['per_query']['query_count'] == 3
    result2 = compare_rankers(groups, scorer=lambda pairs: (_ for _ in ()).throw(AssertionError()),
                              model_id='mock', cache_path=tmp_path / 'cache.json')
    assert 'per_query' not in result2


def test_invalid_scores_are_rejected():
    with pytest.raises(ValueError, match='missing score'):
        per_query_comparison(sample(), {}, {})


def test_full_split_is_cli_default():
    # Verify the source script's CLI help documents the full-split behavior.
    script = Path(__file__).resolve().parents[1] / 'examples' / '09_prepare_scifact.py'
    result = subprocess.run([sys.executable, str(script), '--help'], capture_output=True, text=True)
    assert result.returncode == 0
    assert 'omitted means the full official split' in ' '.join(result.stdout.split())
