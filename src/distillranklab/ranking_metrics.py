"""Judgment-aware MRR@k and nDCG@k for fixed query candidate pools.

Two explicitly different unjudged policies are supported:
  'nonrelevant': retain unjudged candidates in the ranking, assign gain zero.
  'judged_only': remove unjudged candidates BEFORE ranking and cutoff.
Neither policy establishes the true relevance of unjudged passages.
Queries without a judged positive are excluded from both metric means.
"""

import math
from collections.abc import Mapping, Sequence

from .real_data import QueryGroup


def evaluate_ranking(groups: Sequence[QueryGroup],
                     scores: Mapping[tuple[str, str], float], *,
                     k: int = 10, unjudged: str = 'nonrelevant') -> dict[str, float | int]:
    """Evaluate fixed candidate pools; require one finite score per candidate.

    MRR uses relevance > 0; nDCG uses graded gain 2**relevance - 1.
    Ties are broken by passage_id (ascending), independent of input order.
    IDCG is computed from judged positive grades within the candidate pool.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError('k must be a positive integer')
    if unjudged not in ('nonrelevant', 'judged_only'):
        raise ValueError('unjudged must be nonrelevant or judged_only')
    if not groups:
        raise ValueError('groups must be nonempty')
    if not isinstance(scores, Mapping):
        raise ValueError('scores must be a mapping')

    seen = set()
    mrr_sum = ndcg_sum = 0.0
    eligible = excluded = judged = unjudged_count = candidates = 0
    for group in groups:
        if group.query_id in seen:
            raise ValueError('duplicate query ID')
        seen.add(group.query_id)
        if not group.candidates:
            raise ValueError('query groups must have candidates')
        entries = []
        passage_ids = set()
        positive_grades = []
        for candidate in group.candidates:
            if candidate.passage_id in passage_ids:
                raise ValueError('duplicate passage ID within query')
            passage_ids.add(candidate.passage_id)
            key = (group.query_id, candidate.passage_id)
            if key not in scores:
                raise ValueError(f'missing score for {key!r}')
            raw = scores[key]
            if isinstance(raw, bool):
                raise ValueError('scores must be finite real numbers')
            try:
                value = float(raw)
            except (ValueError, TypeError, OverflowError) as exc:
                raise ValueError('scores must be finite real numbers') from exc
            if not math.isfinite(value):
                raise ValueError('scores must be finite real numbers')
            relevance = candidate.relevance
            if relevance is not None and (not isinstance(relevance, int) or isinstance(relevance, bool)
                                          or relevance < 0):
                raise ValueError('relevance must be a nonnegative integer or None')
            candidates += 1
            if relevance is None:
                unjudged_count += 1
                if unjudged == 'judged_only':
                    continue
            else:
                judged += 1
                if relevance > 0:
                    positive_grades.append(relevance)
            entries.append((value, candidate.passage_id, relevance or 0))

        if not positive_grades:
            excluded += 1
            continue
        eligible += 1
        ranked = sorted(entries, key=lambda item: (-item[0], item[1]))[:k]
        for rank, (_, _, rel) in enumerate(ranked, 1):
            if rel > 0:
                mrr_sum += 1.0 / rank
                break
        dcg = sum(math.expm1(rel * math.log(2)) / math.log2(rank + 1)
                  for rank, (_, _, rel) in enumerate(ranked, 1) if rel > 0)
        ideal = sum(math.expm1(rel * math.log(2)) / math.log2(rank + 1)
                    for rank, rel in enumerate(sorted(positive_grades, reverse=True)[:k], 1))
        ndcg_sum += dcg / ideal

    return {
        'mrr': mrr_sum / eligible if eligible else 0.0,
        'ndcg': ndcg_sum / eligible if eligible else 0.0,
        'eligible_queries': eligible,
        'excluded_no_positive': excluded,
        'total_queries': len(groups),
        'candidates': candidates,
        'judged_candidates': judged,
        'unjudged_candidates': unjudged_count,
    }
