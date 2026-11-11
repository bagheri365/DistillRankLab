"""Offline fixed-pool benchmark comparison regression tests."""
import json

import pytest

from distillranklab.benchmark import (bm25_order_scores, compare_rankers,
                                      load_benchmark, summarize_ranking)
from distillranklab.real_data import Candidate, QueryGroup


def groups():
    return (
        QueryGroup('q1', 'apple', (Candidate('b', 'wrong', None),
                                   Candidate('a', 'right', 1))),
        QueryGroup('q2', 'banana', (Candidate('c', 'unjudged', None),
                                    Candidate('d', 'also unjudged', None))),
    )


def test_bm25_respects_candidate_order():
    scores = bm25_order_scores(groups())
    assert scores[('q1', 'b')] > scores[('q1', 'a')]
    assert scores[('q2', 'c')] > scores[('q2', 'd')]


def test_conditional_vs_all_query_denominators():
    result = summarize_ranking(groups(), bm25_order_scores(groups()))
    assert result['total_queries'] == 2
    assert result['queries_with_retrieved_positive'] == 1
    assert result['queries_without_retrieved_positive'] == 1
    assert result['positive_retrieval_coverage'] == 0.5
    assert result['mrr_at_k_conditional'] == 0.5
    assert result['mrr_at_k_all_queries'] == 0.25


def test_teacher_improves_only_eligible_query(tmp_path):
    calls = []
    def scorer(pairs):
        calls.extend(pairs)
        return [10 if passage == 'right' else 0 for _, passage in pairs]
    result = compare_rankers(groups(), scorer=scorer, model_id='mock',
                             cache_path=tmp_path / 'scores.json')
    assert len(calls) == 4
    assert result['bm25']['mrr_at_k_all_queries'] == 0.25
    assert result['teacher']['mrr_at_k_conditional'] == 1
    assert result['teacher']['mrr_at_k_all_queries'] == 0.5
    assert result['teacher']['positive_retrieval_coverage'] == 0.5
    assert result['teacher_scoring_wall_seconds'] >= 0


def test_cache_hit(tmp_path):
    path = tmp_path / 'scores.json'
    compare_rankers(groups(), scorer=lambda pairs: [1] * len(pairs),
                    model_id='mock', cache_path=path)
    def fail(_):
        raise AssertionError('unexpected inference on cache hit')
    compare_rankers(groups(), scorer=fail, model_id='mock', cache_path=path)


def test_judged_only_changes_rank(tmp_path):
    a = summarize_ranking(groups(), bm25_order_scores(groups()), unjudged='nonrelevant')
    b = summarize_ranking(groups(), bm25_order_scores(groups()), unjudged='judged_only')
    assert a['mrr_at_k_conditional'] == 0.5
    assert b['mrr_at_k_conditional'] == 1.0


def test_no_positive_returns_zeros():
    subset = groups()[1:]
    result = summarize_ranking(subset, bm25_order_scores(subset))
    assert result['mrr_at_k_all_queries'] == 0
    assert result['ndcg_at_k_all_queries'] == 0
    assert result['positive_retrieval_coverage'] == 0


def test_k_cutoff():
    result = summarize_ranking(groups(), bm25_order_scores(groups()), k=1)
    assert result['mrr_at_k_conditional'] == 0


def test_missing_score_rejected():
    with pytest.raises(ValueError, match='missing score'):
        summarize_ranking(groups(), {})


def make_dataset(tmp_path, *, positives=1):
    (tmp_path / 'queries.tsv').write_text('q1\tquery one\nq2\tquery two\n')
    (tmp_path / 'passages.tsv').write_text('a\talpha\nb\tbeta\n')
    (tmp_path / 'candidates.tsv').write_text('q1\ta\nq2\tb\n')
    (tmp_path / 'qrels.tsv').write_text('q1\ta\t1\n' if positives else '')
    manifest = dict(dataset='BEIR SciFact', split='test', positive_injection=False,
                    query_count=2, candidate_count=2,
                    queries_with_retrieved_positive=positives)
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    return manifest


def test_load_benchmark(tmp_path):
    make_dataset(tmp_path)
    loaded, manifest = load_benchmark(tmp_path)
    assert len(loaded) == 2
    assert manifest['query_count'] == 2


@pytest.mark.parametrize('field,value', [
    ('dataset', 'other'), ('split', 'dev'), ('positive_injection', True),
    ('query_count', 3), ('candidate_count', 3),
    ('queries_with_retrieved_positive', 0), ('query_count', True),
])
def test_manifest_inconsistency_rejected(tmp_path, field, value):
    manifest = make_dataset(tmp_path)
    manifest[field] = value
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_benchmark(tmp_path)
