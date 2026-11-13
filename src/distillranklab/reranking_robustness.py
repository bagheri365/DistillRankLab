"""10B: frozen-shortlist descriptive robustness, not inference-time routing."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import numpy as np

from .benchmark import per_query_comparison, summarize_ranking
from .conservative_reranking import rank_scores
from .real_data import QueryGroup

SHORTLIST = ('bm25', 'cosine', 'blend_0.5', 'blend_0.75')


def top_ids(groups: Sequence[QueryGroup], scores: Mapping[tuple[str, str], float],
            *, k: int = 10) -> dict[str, tuple[str, ...]]:
    """Deterministic tie-breaking by original BM25 order, never qrels."""
    if k < 1:
        raise ValueError('k must be positive')
    result = {}
    for g in groups:
        ids = [c.passage_id for c in g.candidates]
        values = [scores[g.query_id, pid] for pid in ids]
        if any(not math.isfinite(v) for v in values):
            raise ValueError('nonfinite score')
        order = sorted(range(len(ids)), key=lambda i: (-values[i], i))
        result[g.query_id] = tuple(ids[i] for i in order[:k])
    return result


def _rank_bin(rank):
    if rank is None:
        return 'not_retrieved'
    return 'top_1' if rank == 1 else 'ranks_2_10' if rank <= 10 else 'ranks_11_plus'


def positive_rank_transitions(groups, base, challenger):
    """Judgment-dependent *post-hoc* diagnostic, not deployment routing."""
    n = max(len(g.candidates) for g in groups)
    a, b = top_ids(groups, base, k=n), top_ids(groups, challenger, k=n)
    counts, rows = {}, []
    for g in sorted(groups, key=lambda x: x.query_id):
        positives = {c.passage_id for c in g.candidates if c.relevance is not None and c.relevance > 0}
        ra = next((i for i, pid in enumerate(a[g.query_id], 1) if pid in positives), None)
        rb = next((i for i, pid in enumerate(b[g.query_id], 1) if pid in positives), None)
        transition = f'{_rank_bin(ra)} -> {_rank_bin(rb)}'
        counts[transition] = counts.get(transition, 0) + 1
        rows.append({'query_id': g.query_id, 'bm25_positive_rank': ra,
                     'challenger_positive_rank': rb, 'transition': transition})
    return {'counts': dict(sorted(counts.items())), 'queries': rows}


def paired_audit(groups, baseline, challenger, *, top_n=5):
    """Paired per-query deltas and leave-one-out influence (not a CI)."""
    if top_n < 1 or len(groups) < 2:
        raise ValueError('top_n >= 1 and >= 2 queries required')
    rows = per_query_comparison(groups, baseline, challenger, k=10)['queries']
    delta = np.asarray([r['delta_ndcg_at_k'] for r in rows], dtype=float)
    loo = (delta.sum() - delta) / (len(rows) - 1)
    return {
        'query_count': len(rows), 'mean_delta_ndcg_at_10': float(delta.mean()),
        'improved': int(np.count_nonzero(delta > 1e-12)),
        'regressed': int(np.count_nonzero(delta < -1e-12)),
        'unchanged': int(np.count_nonzero(np.abs(delta) <= 1e-12)),
        'leave_one_out_mean_delta_min': float(loo.min()),
        'leave_one_out_mean_delta_max': float(loo.max()),
        'leave_one_out_positive_count': int(np.count_nonzero(loo > 0)),
        'top_gains': [{'query_id': r['query_id'], 'delta_ndcg_at_10': r['delta_ndcg_at_k']}
                      for r in sorted(rows, key=lambda r: (-r['delta_ndcg_at_k'], r['query_id']))[:top_n]],
        'top_losses': [{'query_id': r['query_id'], 'delta_ndcg_at_10': r['delta_ndcg_at_k']}
                       for r in sorted(rows, key=lambda r: (r['delta_ndcg_at_k'], r['query_id']))[:top_n]],
        'per_query': [{'query_id': r['query_id'], 'delta_ndcg_at_10': r['delta_ndcg_at_k'],
                       'delta_mrr_at_10': r['delta_mrr_at_k']} for r in rows],
    }


def shortlist_scores(groups, matrices):
    """Exactly four 10A strategies; no parameter selection."""
    return {'bm25': rank_scores(groups, matrices, method='bm25'),
            'cosine': rank_scores(groups, matrices, method='cosine'),
            'blend_0.5': rank_scores(groups, matrices, method='blend', alpha=.5),
            'blend_0.75': rank_scores(groups, matrices, method='blend', alpha=.75)}


def audit_pool(groups, matrices, *, cutoffs=(20, 50, 100)):
    """Truncation changes candidate coverage as well as reranking opportunities."""
    if (not groups or len(groups) != len(matrices) or not cutoffs or
            len(set(cutoffs)) != len(cutoffs) or
            any(type(n) is not int or n < 2 for n in cutoffs)):
        raise ValueError('invalid groups or cutoffs')
    result = {}
    for cutoff in cutoffs:
        subset = tuple(QueryGroup(g.query_id, g.query, g.candidates[:cutoff]) for g in groups)
        if any(len(g.candidates) < 2 for g in subset):
            raise ValueError('candidate pool too small')
        features = tuple(np.asarray(x)[:cutoff] for x in matrices)
        scores = shortlist_scores(subset, features)
        top = {name: top_ids(subset, s, k=1) for name, s in scores.items()}
        result[str(cutoff)] = {
            'candidate_count': sum(len(g.candidates) for g in subset),
            'models': {name: summarize_ranking(subset, s, k=10) for name, s in scores.items()},
            'top1_agreement_with_bm25': {
                name: sum(top[name][g.query_id] == top['bm25'][g.query_id] for g in subset)
                for name in SHORTLIST if name != 'bm25'},
        }
    return result


def shortlist_audit(groups, matrices):
    scores = shortlist_scores(groups, matrices)
    top = {name: top_ids(groups, s, k=10) for name, s in scores.items()}
    result = {'models': {name: summarize_ranking(groups, s, k=10) for name, s in scores.items()},
              'top1_agreement': {}, 'top10_overlap_mean': {},
              'comparisons_vs_bm25': {}, 'comparisons_vs_cosine': {}}
    for name in SHORTLIST[1:]:
        result['top1_agreement'][name] = sum(
            top[name][g.query_id][0] == top['bm25'][g.query_id][0] for g in groups)
        result['top10_overlap_mean'][name] = sum(
            len(set(top[name][g.query_id]) & set(top['bm25'][g.query_id]))
            for g in groups) / len(groups)
        result['comparisons_vs_bm25'][name] = paired_audit(groups, scores['bm25'], scores[name])
    for name in ('blend_0.5', 'blend_0.75'):
        result['comparisons_vs_cosine'][name] = paired_audit(groups, scores['cosine'], scores[name])
    result['positive_rank_transitions_bm25_to_blend_0.75'] = positive_rank_transitions(
        groups, scores['bm25'], scores['blend_0.75'])
    return result
