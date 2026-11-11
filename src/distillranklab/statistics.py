"""Paired query-level uncertainty estimates for fixed-pool ranking comparisons.

Bootstrap percentile intervals describe sampling variability across queries.
A paired sign-flip randomization test evaluates the sharp null of exchangeable
ranker labels within each query. Neither procedure fixes incomplete qrels.
"""

import math
from collections.abc import Mapping

import numpy as np

METRICS = ('mrr', 'ndcg')


def paired_statistics(diagnostics: Mapping, *, replicates: int = 10000,
                      seed: int = 0, confidence: float = 0.95) -> dict:
    """Analyze all-query paired metrics, including zero-credit retrieval misses.

    The bootstrap resamples queries with replacement. The two-sided sign-flip
    Monte Carlo p-value uses a +1 correction and includes the observed sample.
    """
    if type(replicates) is not int or replicates < 100:
        raise ValueError('replicates must be an integer >= 100')
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError('confidence must be between 0 and 1')
    if not isinstance(diagnostics, Mapping) or not isinstance(diagnostics.get('queries'), list):
        raise ValueError('expected per-query diagnostics with a queries list')
    rows = diagnostics['queries']
    if not rows or diagnostics.get('query_count') != len(rows):
        raise ValueError('query_count mismatch or empty query list')
    ids = [row.get('query_id') for row in rows]
    if any(not isinstance(q, str) or not q for q in ids) or len(set(ids)) != len(ids):
        raise ValueError('missing or duplicate query IDs')
    arrays = {}
    for metric in METRICS:
        b_key, t_key = f'bm25_{metric}_at_k', f'teacher_{metric}_at_k'
        try:
            baseline = np.asarray([row[b_key] for row in rows], dtype=float)
            teacher = np.asarray([row[t_key] for row in rows], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'invalid {metric} values') from exc
        if (not np.all(np.isfinite(baseline)) or not np.all(np.isfinite(teacher))
                or np.any((baseline < 0) | (baseline > 1))
                or np.any((teacher < 0) | (teacher > 1))):
            raise ValueError(f'{metric} scores must be finite and within [0, 1]')
        delta_key = f'delta_{metric}_at_k'
        if any(delta_key not in row or not isinstance(row[delta_key], (int, float))
               or not math.isfinite(row[delta_key]) for row in rows):
            raise ValueError(f'missing or invalid {delta_key}')
        recorded = np.array([row[delta_key] for row in rows], dtype=float)
        if not np.allclose(recorded, teacher - baseline, rtol=0, atol=1e-10):
            raise ValueError(f'inconsistent {delta_key}')
        arrays[metric] = teacher - baseline

    rng = np.random.default_rng(seed)
    n = len(rows)
    results = {}
    # Draw indices/signs in chunks to bound memory even with large replicate counts.
    for metric, differences in arrays.items():
        boot = np.empty(replicates)
        exceed = 0
        observed = float(differences.mean())
        for start in range(0, replicates, 1024):
            stop = min(start + 1024, replicates)
            count = stop - start
            indices = rng.integers(0, n, size=(count, n))
            boot[start:stop] = differences[indices].mean(axis=1)
            signs = rng.integers(0, 2, size=(count, n), dtype=np.int8) * 2 - 1
            permuted = (signs * differences).mean(axis=1)
            exceed += int(np.count_nonzero(np.abs(permuted) >= abs(observed) - 1e-15))
        low, high = np.quantile(boot, [(1 - confidence) / 2, (1 + confidence) / 2])
        results[metric] = {
            'mean_teacher_minus_bm25': observed,
            'bootstrap_percentile_ci': [float(low), float(high)],
            'paired_sign_flip_p_two_sided': (exceed + 1) / (replicates + 1),
        }
    return {
        'query_count': n, 'k': diagnostics.get('k'),
        'unjudged': diagnostics.get('unjudged'),
        'replicates': replicates, 'seed': seed, 'confidence': confidence,
        'sampling_unit': 'query',
        'method': 'paired query bootstrap percentile CI; paired Monte Carlo sign-flip test',
        'scope': 'fixed BM25 candidate pool; all queries; missing retrieved positives score zero',
        'limitations': 'Unjudged candidates treated according to input policy; not full-corpus BEIR; exploratory multiple metrics',
        'metrics': results,
    }
