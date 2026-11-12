"""Reproducibility and incomplete-qrels audit for fixed-pool ranking.

Judged-only is a sensitivity analysis, not a correction or a new ground truth.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math

import numpy as np

from .benchmark import per_query_comparison, summarize_ranking
from .real_data import QueryGroup


def candidate_fingerprint(groups: Sequence[QueryGroup]) -> str:
    """Order-sensitive SHA256 of query IDs, candidate IDs, and judgments."""
    if not groups:
        raise ValueError('empty candidate pool')
    seen = set()
    records = []
    for group in sorted(groups, key=lambda g: g.query_id):
        if group.query_id in seen or not group.candidates:
            raise ValueError('duplicate query or empty candidates')
        seen.add(group.query_id)
        ids = [c.passage_id for c in group.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError('duplicate candidate')
        records.append([group.query_id, [[c.passage_id, c.relevance] for c in group.candidates]])
    return hashlib.sha256(json.dumps(records, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def validate_score_pool(groups: Sequence[QueryGroup], scores: Mapping, name: str) -> dict:
    """Reject missing/extra candidate scores, duplicate IDs and nonfinite values."""
    if not isinstance(scores, Mapping):
        raise ValueError(f'{name}: scores must be a mapping')
    fingerprint = candidate_fingerprint(groups)
    expected = {(g.query_id, c.passage_id) for g in groups for c in g.candidates}
    actual = set(scores)
    if expected != actual:
        raise ValueError(f'{name}: candidate score pool mismatch: {len(expected - actual)} missing, {len(actual - expected)} extra')
    for key, value in scores.items():
        if isinstance(value, (bool, str)):
            raise ValueError(f'{name}: invalid score at {key!r}')
        try:
            number = float(value)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f'{name}: invalid score at {key!r}') from exc
        if not math.isfinite(number):
            raise ValueError(f'{name}: nonfinite score at {key!r}')
    return {'candidate_pool_sha256': fingerprint, 'query_count': len(groups), 'candidate_count': len(expected)}


def _check_aggregate(diagnostics: dict, baseline: dict, challenger: dict, tolerance: float = 1e-10):
    rows = diagnostics['queries']
    for metric in ('mrr', 'ndcg'):
        for side, summary in (('bm25', baseline), ('teacher', challenger)):
            actual = float(np.mean([r[f'{side}_{metric}_at_k'] for r in rows]))
            expected = summary[f'{metric}_at_k_all_queries']
            if not math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance):
                raise ValueError(f'per-query/aggregate mismatch for {side} {metric}')
        delta = float(np.mean([r[f'delta_{metric}_at_k'] for r in rows]))
        expected_delta = challenger[f'{metric}_at_k_all_queries'] - baseline[f'{metric}_at_k_all_queries']
        if not math.isclose(delta, expected_delta, rel_tol=0, abs_tol=tolerance):
            raise ValueError(f'per-query delta mismatch for {metric}')


def audit_pair(groups: Sequence[QueryGroup], baseline: Mapping, challenger: Mapping,
               *, baseline_name: str, challenger_name: str) -> dict:
    """Check both policies, aggregate agreement and distribution of paired deltas."""
    pool = validate_score_pool(groups, baseline, baseline_name)
    validate_score_pool(groups, challenger, challenger_name)
    policies = {}
    for policy in ('nonrelevant', 'judged_only'):
        b = summarize_ranking(groups, baseline, k=10, unjudged=policy)
        c = summarize_ranking(groups, challenger, k=10, unjudged=policy)
        diagnostics = per_query_comparison(groups, baseline, challenger, k=10, unjudged=policy)
        _check_aggregate(diagnostics, b, c)
        rows = diagnostics['queries']
        distributions = {}
        for metric in ('mrr', 'ndcg'):
            delta = np.array([r[f'delta_{metric}_at_k'] for r in rows], dtype=float)
            distributions[metric] = {
                'mean_delta': float(delta.mean()),
                'median_delta': float(np.median(delta)),
                'min_delta': float(delta.min()),
                'max_delta': float(delta.max()),
                'sample_sd': float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
                'improved_queries': int(np.count_nonzero(delta > 1e-12)),
                'regressed_queries': int(np.count_nonzero(delta < -1e-12)),
                'unchanged_queries': int(np.count_nonzero(np.abs(delta) <= 1e-12)),
                'quantiles_05_25_75_95': [float(x) for x in np.quantile(delta, [0.05, 0.25, 0.75, 0.95])],
            }
        policies[policy] = {'baseline': b, 'challenger': c,
                            'per_query_aggregate_verified': True, 'delta_distribution': distributions}
    return {'baseline_name': baseline_name, 'challenger_name': challenger_name,
            'pool': pool, 'policies': policies,
            'interpretation': ('Judged-only removes unjudged candidates before ranking; '
                               'it is a sensitivity scenario, not evidence those candidates are irrelevant.')}
