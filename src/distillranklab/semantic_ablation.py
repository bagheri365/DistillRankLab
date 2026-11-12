"""Feature-block ablation and paired fixed-pool query comparisons.

Masking features holds the 10-coordinate head implementation constant. Masked
weights cannot affect scores; effective trainable dimensions are 8, 2, or 10.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .benchmark import per_query_comparison, summarize_ranking
from .real_data import QueryGroup
from .statistics import paired_statistics
from .text_student import TEXT_FEATURE_NAMES, fit_text_head, predict_text_head

FEATURE_BLOCKS = {'lexical_only': tuple(range(8)),
                  'semantic_only': (8, 9),
                  'combined': tuple(range(10))}


def masked_features(matrices: Sequence[np.ndarray], block: str) -> list[np.ndarray]:
    """Return independent arrays with all nonselected coordinates exactly zero."""
    if block not in FEATURE_BLOCKS:
        raise ValueError('unknown feature block')
    keep = FEATURE_BLOCKS[block]
    output = []
    for matrix in matrices:
        x = np.asarray(matrix, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(TEXT_FEATURE_NAMES) or not np.isfinite(x).all():
            raise ValueError('invalid feature matrix')
        masked = np.zeros_like(x)
        masked[:, keep] = x[:, keep]
        output.append(masked)
    return output


def run_feature_ablation(fit: Sequence[QueryGroup], validation: Sequence[QueryGroup],
                        fit_features: Sequence[np.ndarray], validation_features: Sequence[np.ndarray],
                        teacher: Mapping[tuple[str, str], float], *, steps: int = 120,
                        seed: int = 42) -> tuple[dict, dict[str, dict]]:
    """Fit three heads on identical queries and return reports and scores."""
    if not fit or not validation or len(fit) != len(fit_features) or len(validation) != len(validation_features):
        raise ValueError('groups and features mismatch')
    report, predictions = {}, {}
    for block, coordinates in FEATURE_BLOCKS.items():
        model = fit_text_head(fit, masked_features(fit_features, block), teacher,
                              steps=steps, learning_rate=0.05, temperature=2.0,
                              hard_weight=0.5, seed=seed)
        scores = predict_text_head(validation, masked_features(validation_features, block), model)
        report[block] = {'metrics': summarize_ranking(validation, scores, k=10),
                         'active_feature_names': [TEXT_FEATURE_NAMES[i] for i in coordinates],
                         'effective_parameters': len(coordinates),
                         'implementation_parameters': len(TEXT_FEATURE_NAMES)}
        predictions[block] = scores
    return report, predictions


def compare_paired(groups: Sequence[QueryGroup], baseline: Mapping,
                   challenger: Mapping, *, baseline_name: str,
                   challenger_name: str, replicates: int = 10000, seed: int = 42) -> dict:
    """Paired query uncertainty, with explicit names (legacy statistic keys retained)."""
    diagnostics = per_query_comparison(groups, baseline, challenger, k=10)
    stats = paired_statistics(diagnostics, replicates=replicates, seed=seed)
    return {'baseline': baseline_name, 'challenger': challenger_name,
            'difference': f'{challenger_name} minus {baseline_name}',
            'query_count': stats['query_count'], 'replicates': replicates,
            'sampling_unit': 'query', 'metrics': {
                metric: {'mean_challenger_minus_baseline': result['mean_teacher_minus_bm25'],
                         'bootstrap_percentile_ci': result['bootstrap_percentile_ci'],
                         'paired_sign_flip_p_two_sided': result['paired_sign_flip_p_two_sided']}
                for metric, result in stats['metrics'].items()},
            'limitation': 'Exploratory reused validation; unjudged treated nonrelevant; no multiple-test adjustment'}
