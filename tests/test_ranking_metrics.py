"""Offline ranking metric tests with explicit incomplete-qrels assumptions."""
import math
from pathlib import Path

import pytest

from distillranklab.ranking_metrics import evaluate_ranking
from distillranklab.real_data import Candidate, QueryGroup, load_groups


def group(qid, *relevances):
    return QueryGroup(qid, 'query', tuple(
        Candidate(f'p{i}', f'passage {i}', rel) for i, rel in enumerate(relevances)))


def score(group_, *values):
    return {(group_.query_id, c.passage_id): value
            for c, value in zip(group_.candidates, values)}


def test_perfect_binary():
    g = group('q', 1, 0, None)
    result = evaluate_ranking([g], score(g, 3, 2, 1))
    assert result['mrr'] == result['ndcg'] == 1
    assert result['unjudged_candidates'] == 1


def test_relevant_second():
    g = group('q', 0, 1)
    result = evaluate_ranking([g], score(g, 2, 1))
    assert result['mrr'] == 0.5
    assert result['ndcg'] == pytest.approx(1 / math.log2(3))


def test_cutoff_excludes_relevant():
    g = group('q', 0, 1)
    result = evaluate_ranking([g], score(g, 2, 1), k=1)
    assert result['mrr'] == result['ndcg'] == 0


def test_graded_ndcg():
    g = group('q', 1, 2, 0)
    result = evaluate_ranking([g], score(g, 3, 2, 1))
    dcg = 1 + 3 / math.log2(3)
    ideal = 3 + 1 / math.log2(3)
    assert result['mrr'] == 1
    assert result['ndcg'] == pytest.approx(dcg / ideal)


def test_unjudged_policies_are_different():
    g = group('q', None, 1, 0)
    values = score(g, 9, 8, 7)
    zero = evaluate_ranking([g], values, k=2, unjudged='nonrelevant')
    filtered = evaluate_ranking([g], values, k=2, unjudged='judged_only')
    assert zero['mrr'] == 0.5
    assert filtered['mrr'] == 1
    assert filtered['ndcg'] == 1


def test_no_positive_excluded_from_denominator():
    a, b = group('a', 1, 0), group('b', 0, None)
    result = evaluate_ranking([a, b], {**score(a, 2, 1), **score(b, 2, 1)})
    assert result['mrr'] == result['ndcg'] == 1
    assert result['eligible_queries'] == 1
    assert result['excluded_no_positive'] == 1
    assert result['total_queries'] == 2


def test_all_queries_without_positives():
    g = group('q', 0, None)
    result = evaluate_ranking([g], score(g, 1, 2))
    assert result['mrr'] == result['ndcg'] == 0
    assert result['eligible_queries'] == 0


def test_macro_average_not_candidate_weighted():
    a, b = group('a', 1, 0), group('b', 0, 1, 0, 0)
    result = evaluate_ranking([a, b], {**score(a, 2, 1), **score(b, 4, 3, 2, 1)})
    assert result['mrr'] == 0.75


def test_ties_break_by_passage_id_not_input_order():
    a = QueryGroup('q', 'q', (Candidate('z', 'z', 1), Candidate('a', 'a', 0)))
    b = QueryGroup('q', 'q', tuple(reversed(a.candidates)))
    values = {('q', 'z'): 1.0, ('q', 'a'): 1.0}
    assert evaluate_ranking([a], values) == evaluate_ranking([b], values)
    assert evaluate_ranking([a], values)['mrr'] == 0.5


def test_k_larger_than_candidates():
    g = group('q', 0, 1)
    assert evaluate_ranking([g], score(g, 2, 1), k=10)['mrr'] == 0.5


@pytest.mark.parametrize('bad', [0, -1, 2.5, True])
def test_bad_k(bad):
    g = group('q', 1)
    with pytest.raises(ValueError, match='k must'):
        evaluate_ranking([g], score(g, 1), k=bad)


def test_bad_policy():
    g = group('q', 1)
    with pytest.raises(ValueError, match='unjudged'):
        evaluate_ranking([g], score(g, 1), unjudged='ignore')


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf'), True, 'oops'])
def test_bad_score(bad):
    g = group('q', 1)
    with pytest.raises(ValueError, match='finite'):
        evaluate_ranking([g], score(g, bad))


def test_missing_score():
    g = group('q', 1)
    with pytest.raises(ValueError, match='missing score'):
        evaluate_ranking([g], {})


def test_empty_groups():
    with pytest.raises(ValueError, match='nonempty'):
        evaluate_ranking([], {})


def test_duplicate_query_ids():
    g = group('q', 1)
    with pytest.raises(ValueError, match='duplicate query'):
        evaluate_ranking([g, g], score(g, 1))


def test_duplicate_passage_ids():
    g = QueryGroup('q', 'q', (Candidate('p', 'a', 1), Candidate('p', 'b', 0)))
    with pytest.raises(ValueError, match='duplicate passage'):
        evaluate_ranking([g], {('q', 'p'): 1})


def test_bad_relevance():
    g = group('q', -1)
    with pytest.raises(ValueError, match='relevance'):
        evaluate_ranking([g], score(g, 1))


def test_fixture_smoke():
    base = Path(__file__).parent / 'fixtures' / 'tiny_ranking'
    groups = load_groups(*(base / name for name in
                           ('queries.tsv', 'passages.tsv', 'candidates.tsv', 'qrels.tsv')))
    scores = {(g.query_id, c.passage_id): float(-i)
              for g in groups for i, c in enumerate(g.candidates)}
    result = evaluate_ranking(groups, scores)
    assert result['total_queries'] == 5
    assert result['candidates'] == 10
    assert 0 <= result['mrr'] <= 1
    assert 0 <= result['ndcg'] <= 1
