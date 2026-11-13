"""Exploratory fit-query audit of cross-encoder ranking vs sparse relevance labels.

No student is trained. Unjudged candidates are *not* known negatives.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import math

import numpy as np

from .benchmark import bm25_order_scores, per_query_comparison, summarize_ranking
from .evaluation_audit import validate_score_pool
from .real_data import QueryGroup


def _rank(scores: np.ndarray) -> np.ndarray:
    """Stable descending ranks, 1-based; original BM25 order breaks ties."""
    order = np.argsort(-scores, kind='stable')
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def _bucket(rank: int | None) -> str:
    if rank is None:
        return 'no_retrieved_positive'
    return 'top_1' if rank == 1 else 'ranks_2_10' if rank <= 10 else 'ranks_11_100'


def teacher_relevance_alignment(groups: Sequence[QueryGroup], teacher: Mapping,
                                *, k: int = 10, examples: int = 5) -> dict:
    """Compare teacher/BM25 on the same fit-query pool, without injecting positives.

    Positive-vs-other margins compare to *unjudged or judged* other candidates;
    they are NOT positive-vs-verified-negative classification margins.
    """
    if not groups or k < 1 or examples < 0:
        raise ValueError('nonempty groups, positive k and nonnegative examples required')
    validate_score_pool(groups, teacher, 'teacher')
    bm25 = bm25_order_scores(groups)
    comparisons = per_query_comparison(groups, bm25, teacher, k=k)
    per_query = {r['query_id']: r for r in comparisons['queries']}
    rows = []
    for g in sorted(groups, key=lambda g: g.query_id):
        ids = [(g.query_id, c.passage_id) for c in g.candidates]
        scores = np.asarray([teacher[key] for key in ids], dtype=float)
        ranks = _rank(scores)
        positives = [i for i, c in enumerate(g.candidates) if c.relevance is not None and c.relevance > 0]
        judged_zeros = [i for i, c in enumerate(g.candidates) if c.relevance == 0]
        unjudged = sum(c.relevance is None for c in g.candidates)
        best_bm25 = min((i + 1 for i in positives), default=None)
        best_teacher = min((int(ranks[i]) for i in positives), default=None)
        best_positive_score = max((float(scores[i]) for i in positives), default=None)
        other = [i for i in range(len(ids)) if i not in positives]
        margin_to_other = (best_positive_score - float(np.max(scores[other]))
                           if best_positive_score is not None and other else None)
        verified_margin = (best_positive_score - max(float(scores[i]) for i in judged_zeros)
                           if best_positive_score is not None and judged_zeros else None)
        top = int(np.argmin(ranks))
        row = {'query_id': g.query_id, 'candidate_count': len(ids),
               'judged_positive_count': len(positives), 'judged_zero_count': len(judged_zeros),
               'unjudged_count': unjudged,
               'bm25_best_positive_rank': best_bm25, 'teacher_best_positive_rank': best_teacher,
               'bm25_positive_rank_bin': _bucket(best_bm25),
               'teacher_positive_rank_bin': _bucket(best_teacher),
               'teacher_top1_judgment': ('relevant' if g.candidates[top].relevance is not None and g.candidates[top].relevance > 0
                                         else 'judged_nonrelevant' if g.candidates[top].relevance == 0 else 'unjudged'),
               'teacher_top1_id': g.candidates[top].passage_id,
               'bm25_top1_id': g.candidates[0].passage_id,
               'teacher_bm25_top1_disagree': top != 0,
               'positive_margin_vs_best_other_including_unjudged': margin_to_other,
               'positive_margin_vs_best_judged_zero': verified_margin,
               'positive_above_all_other_candidates': margin_to_other is not None and margin_to_other > 0,
               'teacher_improves_best_positive_rank': (best_teacher < best_bm25 if best_bm25 is not None else None),
               'teacher_worsens_best_positive_rank': (best_teacher > best_bm25 if best_bm25 is not None else None),
               'delta_ndcg_at_10': per_query[g.query_id]['delta_ndcg_at_k'],
               'delta_mrr_at_10': per_query[g.query_id]['delta_mrr_at_k']}
        rows.append(row)

    def group_summary(items):
        positives = [r for r in items if r['bm25_best_positive_rank'] is not None]
        return {'queries': len(items), 'queries_with_retrieved_positive': len(positives),
                'mean_delta_ndcg_at_10': float(np.mean([r['delta_ndcg_at_10'] for r in items])) if items else None,
                'teacher_better_positive_rank': sum(r['teacher_improves_best_positive_rank'] is True for r in positives),
                'teacher_worse_positive_rank': sum(r['teacher_worsens_best_positive_rank'] is True for r in positives),
                'teacher_same_positive_rank': sum(r['teacher_best_positive_rank'] == r['bm25_best_positive_rank'] for r in positives)}

    eligible = [r for r in rows if r['bm25_best_positive_rank'] is not None]
    top1_counts = Counter(r['teacher_top1_judgment'] for r in rows)
    disagreement = [r for r in rows if r['teacher_bm25_top1_disagree']]
    return {'queries': len(rows), 'k': k, 'candidate_pool': validate_score_pool(groups, teacher, 'teacher'),
            'bm25': summarize_ranking(groups, bm25, k=k),
            'teacher': summarize_ranking(groups, teacher, k=k),
            'comparison': {key: comparisons[key] for key in ('ndcg_improved_queries', 'ndcg_regressed_queries', 'ndcg_unchanged_queries')},
            'teacher_top1_judgments': dict(top1_counts),
            'teacher_bm25_top1_disagreement': len(disagreement),
            'positive_rank_summary': group_summary(rows),
            'by_bm25_positive_rank': {name: group_summary([r for r in rows if r['bm25_positive_rank_bin'] == name])
                                      for name in ('top_1', 'ranks_2_10', 'ranks_11_100', 'no_retrieved_positive')},
            'by_teacher_top1_judgment': {name: group_summary([r for r in rows if r['teacher_top1_judgment'] == name])
                                         for name in ('relevant', 'judged_nonrelevant', 'unjudged')},
            'by_top1_disagreement': {name: group_summary([r for r in rows if r['teacher_bm25_top1_disagree'] == value])
                                     for name, value in (('agree', False), ('disagree', True))},
            'judged_zero_margin_available': sum(r['positive_margin_vs_best_judged_zero'] is not None for r in rows),
            'top_gains': sorted(rows, key=lambda r: (-r['delta_ndcg_at_10'], r['query_id']))[:examples],
            'top_losses': sorted(rows, key=lambda r: (r['delta_ndcg_at_10'], r['query_id']))[:examples],
            'per_query': rows,
            'limitations': [
                'FIT queries only: teacher preference is in-sample and not generalization evidence',
                'Unjudged candidates are unknown, not verified negatives; margin to other candidates is not a classification margin',
                'Teacher top1 unjudged is not evidence of an error',
                'No retrieved judged positive does not imply no relevant passage in corpus',
                'Fixed BM25 top-100 pool; no positive injection; no official test evaluation',
                'Descriptive exploratory subgroup comparisons, not confirmatory statistical tests',
            ]}
