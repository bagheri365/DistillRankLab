"""Fixed-pool BM25 versus cross-encoder evaluation for BEIR SciFact.

The exported candidates.tsv is already ordered by the first-stage BM25
retriever. Never re-sort candidates using qrels or inject judged positives.
"""

from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import time

from .ranking_metrics import evaluate_ranking
from .real_data import QueryGroup, load_groups
from .teacher import score_groups

FILES = ('queries.tsv', 'passages.tsv', 'candidates.tsv', 'qrels.tsv')


def bm25_order_scores(groups: Sequence[QueryGroup]) -> dict[tuple[str, str], float]:
    """Reconstruct ranking from the original ordered candidate TSV rows.

    These are rank-based scores, NOT the original numerical BM25 scores.
    """
    return {(group.query_id, candidate.passage_id): float(-rank)
            for group in groups for rank, candidate in enumerate(group.candidates)}


def summarize_ranking(groups: Sequence[QueryGroup],
                      scores: Mapping[tuple[str, str], float], *,
                      k: int = 10, unjudged: str = 'nonrelevant') -> dict:
    """Report conditional and all-query means, counting missed positives as 0.

    A candidate pool with no judged positive is counted as a zero on all-query
    metrics. This is still a *fixed-pool* evaluation, not full-corpus BEIR.
    """
    conditional = evaluate_ranking(groups, scores, k=k, unjudged=unjudged)
    n = conditional['total_queries']
    eligible = conditional['eligible_queries']
    return {
        'mrr_at_k_conditional': conditional['mrr'],
        'ndcg_at_k_conditional': conditional['ndcg'],
        'mrr_at_k_all_queries': conditional['mrr'] * eligible / n,
        'ndcg_at_k_all_queries': conditional['ndcg'] * eligible / n,
        'total_queries': n,
        'queries_with_retrieved_positive': eligible,
        'queries_without_retrieved_positive': conditional['excluded_no_positive'],
        'positive_retrieval_coverage': eligible / n,
        'candidate_count': conditional['candidates'],
        'judged_candidates': conditional['judged_candidates'],
        'unjudged_candidates': conditional['unjudged_candidates'],
    }


def load_benchmark(directory: str | Path) -> tuple[tuple[QueryGroup, ...], dict]:
    """Read prepared SciFact TSVs and check manifest consistency."""
    directory = Path(directory)
    groups = load_groups(*(directory / name for name in FILES))
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if not isinstance(manifest, dict) or manifest.get('dataset') != 'BEIR SciFact':
        raise ValueError('expected BEIR SciFact manifest')
    if manifest.get('positive_injection') is not False:
        raise ValueError('benchmark must have no positive injection')
    count = sum(len(g.candidates) for g in groups)
    positives = sum(any(c.relevance is not None and c.relevance > 0
                        for c in g.candidates) for g in groups)
    for key, expected in (('query_count', len(groups)),
                          ('candidate_count', count),
                          ('queries_with_retrieved_positive', positives)):
        value = manifest.get(key)
        if type(value) is not int or value != expected:
            raise ValueError(f'manifest mismatch: {key}')
    if manifest.get('split') not in ('train', 'test'):
        raise ValueError('invalid benchmark split')
    return groups, manifest


def compare_rankers(groups: Sequence[QueryGroup], *, scorer,
                    model_id: str, cache_path: str | Path,
                    batch_size: int = 8, k: int = 10,
                    unjudged: str = 'nonrelevant', per_query: bool = False) -> dict:
    """Compare identical fixed candidate pools and report scoring wall time.

    Time includes cache I/O and any model inference, but excludes model loading.
    A warm cache therefore measures cache retrieval, not inference speed.
    """
    baseline = summarize_ranking(groups, bm25_order_scores(groups),
                                 k=k, unjudged=unjudged)
    start = time.perf_counter()
    teacher_scores = score_groups(groups, scorer=scorer, model_id=model_id,
                                  cache_path=cache_path, batch_size=batch_size)
    seconds = time.perf_counter() - start
    if not math.isfinite(seconds) or seconds < 0:
        raise RuntimeError('invalid timing measurement')
    teacher = summarize_ranking(groups, teacher_scores, k=k, unjudged=unjudged)
    result = {'k': k, 'unjudged': unjudged, 'model_id': model_id,
              'bm25': baseline, 'teacher': teacher,
              'teacher_scoring_wall_seconds': seconds,
              'timing_note': 'Excludes model loading; includes cache I/O; warm-cache time is not inference latency'}
    if per_query:
        result['per_query'] = per_query_comparison(
            groups, bm25_order_scores(groups), teacher_scores, k=k, unjudged=unjudged)
    return result



def per_query_comparison(groups: Sequence[QueryGroup],
                         bm25_scores: Mapping[tuple[str, str], float],
                         teacher_scores: Mapping[tuple[str, str], float], *,
                         k: int = 10, unjudged: str = 'nonrelevant') -> dict:
    """Query-level fixed-pool outcomes; zero for queries with no retrieved positive.

    The two rankers must score the identical candidate pool. Query IDs are sorted
    for stable output. Per-query values are comparable with all-query aggregates.
    """
    rows = []
    for group in sorted(groups, key=lambda g: g.query_id):
        b = summarize_ranking((group,), bm25_scores, k=k, unjudged=unjudged)
        t = summarize_ranking((group,), teacher_scores, k=k, unjudged=unjudged)
        b_mrr, t_mrr = b['mrr_at_k_all_queries'], t['mrr_at_k_all_queries']
        b_ndcg, t_ndcg = b['ndcg_at_k_all_queries'], t['ndcg_at_k_all_queries']
        delta = t_ndcg - b_ndcg
        status = 'improved' if delta > 1e-12 else 'regressed' if delta < -1e-12 else 'unchanged'
        rows.append({'query_id': group.query_id,
                     'retrieved_positive': bool(b['queries_with_retrieved_positive']),
                     'bm25_mrr_at_k': b_mrr, 'teacher_mrr_at_k': t_mrr,
                     'delta_mrr_at_k': t_mrr - b_mrr,
                     'bm25_ndcg_at_k': b_ndcg, 'teacher_ndcg_at_k': t_ndcg,
                     'delta_ndcg_at_k': delta, 'ndcg_outcome': status})
    return {'k': k, 'unjudged': unjudged, 'query_count': len(rows),
            'ndcg_improved_queries': sum(r['ndcg_outcome'] == 'improved' for r in rows),
            'ndcg_regressed_queries': sum(r['ndcg_outcome'] == 'regressed' for r in rows),
            'ndcg_unchanged_queries': sum(r['ndcg_outcome'] == 'unchanged' for r in rows),
            'queries': rows}
