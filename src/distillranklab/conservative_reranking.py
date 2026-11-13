"""10A: inference-only lexical/semantic reranking on a fixed BM25 pool.

No qrels or teacher scores are read by the ranking rules. Original candidate
order supplies a rank prior, not the numerical BM25 retrieval score.
"""
from __future__ import annotations

from collections.abc import Sequence
import math
import numpy as np

from .real_data import QueryGroup


def rank_scores(groups: Sequence[QueryGroup], matrices: Sequence[np.ndarray], *,
                method: str, alpha: float = 0.5, guard_gap: float = 0.05) -> dict[tuple[str, str], float]:
    """Rank by cosine, BM25 order, their convex rank blend, or guarded blend.

    The guard preserves BM25's top passage only when its cosine deficit to the
    semantic winner is at most guard_gap. It never examines relevance labels.
    """
    if method not in ('bm25', 'cosine', 'blend', 'guarded_blend'):
        raise ValueError('unknown ranking method')
    if (not math.isfinite(alpha) or not 0 <= alpha <= 1 or
            not math.isfinite(guard_gap) or guard_gap < 0):
        raise ValueError('invalid blending parameters')
    if not groups or len(groups) != len(matrices) or len({g.query_id for g in groups}) != len(groups):
        raise ValueError('groups/matrices mismatch')
    output = {}
    for group, matrix in zip(groups, matrices):
        x = np.asarray(matrix, dtype=float)
        n = len(group.candidates)
        if x.shape != (n, 10) or n < 2 or not np.isfinite(x).all():
            raise ValueError('invalid feature matrix')
        if len({c.passage_id for c in group.candidates}) != n:
            raise ValueError('duplicate passage IDs')
        cosine = x[:, 8]
        # Stable ranking handles exact cosine ties reproducibly.
        cosine_order = np.argsort(-cosine, kind='stable')
        cosine_ranks = np.empty(n, dtype=int)
        cosine_ranks[cosine_order] = np.arange(n)
        bm25 = 1. - np.arange(n, dtype=float) / (n - 1)
        semantic = 1. - cosine_ranks.astype(float) / (n - 1)
        if method == 'bm25':
            score = bm25
        elif method == 'cosine':
            score = cosine.copy()
        else:
            score = (1. - alpha) * bm25 + alpha * semantic
            if method == 'guarded_blend' and cosine.max() - cosine[0] <= guard_gap:
                # Rank-preserving top-1 guard; other passages retain blend order.
                score = score.copy()
                score[0] = float(score.max() + 1.)
        for candidate, value in zip(group.candidates, score):
            output[group.query_id, candidate.passage_id] = float(value)
    return output


def reranking_suite(groups: Sequence[QueryGroup], matrices: Sequence[np.ndarray], *,
                    alphas: Sequence[float] = (.25, .5, .75),
                    guard_gap: float = .05) -> dict[str, dict[tuple[str, str], float]]:
    """Predeclared variants; no validation-driven parameter selection."""
    if not alphas or len(set(alphas)) != len(alphas):
        raise ValueError('nonempty unique alphas required')
    results = {'bm25': rank_scores(groups, matrices, method='bm25'),
               'cosine': rank_scores(groups, matrices, method='cosine')}
    for alpha in alphas:
        if not math.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError('blend alphas must be strictly between zero and one')
        label = f'{alpha:g}'
        results[f'blend_{label}'] = rank_scores(groups, matrices, method='blend', alpha=alpha)
        results[f'guarded_{label}'] = rank_scores(groups, matrices, method='guarded_blend',
                                                   alpha=alpha, guard_gap=guard_gap)
    return results
