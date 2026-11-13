"""Controlled supervision ablation with an identical frozen encoder and head.

All variants share features, candidate pools, initialization and optimization.
Teacher-only, relevance-only, and hybrid differ solely in supervision weight.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import numpy as np

from .benchmark import summarize_ranking
from .real_data import QueryGroup
from .semantic_ablation import masked_features
from .text_student import TEXT_FEATURE_NAMES, fit_text_head, predict_text_head

SUPERVISION = {'teacher_only': 0.0, 'hybrid': 0.5, 'relevance_only': 1.0}


def cosine_baseline(groups: Sequence[QueryGroup], matrices: Sequence[np.ndarray]) -> dict:
    """Untrained cosine feature: no teacher scores or relevance labels used."""
    if len(groups) != len(matrices):
        raise ValueError('groups and matrices mismatch')
    scores = {}
    for group, x in zip(groups, matrices):
        a = np.asarray(x, dtype=float)
        if a.shape != (len(group.candidates), len(TEXT_FEATURE_NAMES)) or not np.isfinite(a).all():
            raise ValueError('invalid feature matrix')
        for candidate, value in zip(group.candidates, a[:, 8]):
            key = (group.query_id, candidate.passage_id)
            if key in scores:
                raise ValueError('duplicate query/passage key')
            scores[key] = float(value)
    return scores


def supervision_ablation(fit: Sequence[QueryGroup], validation: Sequence[QueryGroup],
                         fit_matrices: Sequence[np.ndarray], validation_matrices: Sequence[np.ndarray],
                         teacher: Mapping[tuple[str, str], float], *, block: str = 'semantic_only',
                         steps: int = 120, seed: int = 42) -> tuple[dict, dict]:
    """Fit matched heads under three supervision modes; return metrics and scores.

    Teacher scores are required for all modes by the shared training API, but
    their coefficient is exactly zero for relevance_only. Hard supervision
    excludes no-positive queries from its normalized gradient.
    """
    if block not in ('semantic_only', 'combined') or not fit or not validation:
        raise ValueError('invalid feature block or empty groups')
    train_x = masked_features(fit_matrices, block)
    val_x = masked_features(validation_matrices, block)
    metrics, predictions = {}, {}
    for name, hard_weight in SUPERVISION.items():
        model = fit_text_head(fit, train_x, teacher, steps=steps, learning_rate=.05,
                              temperature=2.0, hard_weight=hard_weight, seed=seed)
        scores = predict_text_head(validation, val_x, model)
        metrics[name] = {'metrics': summarize_ranking(validation, scores, k=10),
                         'hard_weight': hard_weight, 'effective_head_parameters': 2 if block == 'semantic_only' else 10}
        predictions[name] = scores
    if block == 'semantic_only':
        scores = cosine_baseline(validation, val_x)
        metrics['untrained_cosine'] = {'metrics': summarize_ranking(validation, scores, k=10),
                                     'hard_weight': None, 'effective_head_parameters': 0}
        predictions['untrained_cosine'] = scores
    return metrics, predictions
