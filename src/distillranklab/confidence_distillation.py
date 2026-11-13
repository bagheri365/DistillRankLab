"""9D: label-free teacher-confidence weighting with matched random control.

Weights affect the soft distillation gradient only; hard-label gradients are
unchanged. No relevance labels are consulted when constructing weights.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import numpy as np

from .real_data import QueryGroup
from .real_student import _probabilities
from .text_student import TEXT_FEATURE_NAMES, predict_text_head


def confidence_weights(groups: Sequence[QueryGroup], teacher: Mapping[tuple[str, str], float],
                       *, scheme: str, seed: int = 2026) -> np.ndarray:
    """Return positive, mean-one per-query weights using only teacher and BM25 order.

    margin: teacher top-1 minus top-2 logit, transformed by bounded monotonic
    function; agreement: teacher top-1 equals BM25 top-1; shuffled_margin:
    same weight distribution as margin but permuted across queries.
    """
    if scheme not in ('uniform', 'margin', 'agreement', 'shuffled_margin') or not groups:
        raise ValueError('invalid weighting scheme or empty groups')
    margins, agreements = [], []
    for g in groups:
        if len(g.candidates) < 2:
            raise ValueError('at least two candidates required')
        scores = np.asarray([teacher[g.query_id, c.passage_id] for c in g.candidates], dtype=float)
        if not np.isfinite(scores).all():
            raise ValueError('nonfinite teacher score')
        ordered = np.argsort(-scores, kind='stable')
        margins.append(float(scores[ordered[0]] - scores[ordered[1]]))
        agreements.append(bool(ordered[0] == 0))  # BM25 top-1 is first candidate
    if scheme == 'uniform':
        weights = np.ones(len(groups), dtype=float)
    elif scheme == 'agreement':
        weights = np.where(agreements, 1.5, .5).astype(float)
    else:
        # Bounded confidence avoids letting a single very large logit gap dominate.
        weights = 0.5 + np.asarray(margins) / (1.0 + np.asarray(margins))
        if scheme == 'shuffled_margin':
            weights = weights[np.random.default_rng(seed).permutation(len(weights))]
    return weights / weights.mean()


def fit_weighted_head(groups: Sequence[QueryGroup], matrices: Sequence[np.ndarray],
                      teacher: Mapping[tuple[str, str], float], weights: Sequence[float],
                      *, steps: int = 120, learning_rate: float = .05,
                      temperature: float = 2., hard_weight: float = .5,
                      seed: int = 42) -> dict:
    """Match fit_text_head except for normalized per-query soft-gradient weights."""
    n = len(groups)
    a = np.asarray(weights, dtype=float)
    if (not n or n != len(matrices) or a.shape != (n,) or not np.isfinite(a).all()
            or np.any(a <= 0) or not np.isclose(a.mean(), 1., atol=1e-10)
            or type(steps) is not int or steps < 1 or not math.isfinite(learning_rate)
            or learning_rate <= 0 or not math.isfinite(temperature) or temperature <= 0
            or not math.isfinite(hard_weight) or not 0 <= hard_weight <= 1
            or len({g.query_id for g in groups}) != n):
        raise ValueError('invalid weighted training arguments')
    targets = []
    for g, x in zip(groups, matrices):
        if x.shape != (len(g.candidates), len(TEXT_FEATURE_NAMES)) or not np.isfinite(x).all():
            raise ValueError('invalid feature matrix')
        logits = np.asarray([teacher[g.query_id, c.passage_id] for c in g.candidates], dtype=float)
        if not np.isfinite(logits).all():
            raise ValueError('invalid teacher logits')
        gains = np.asarray([2. ** c.relevance - 1 if c.relevance is not None and c.relevance > 0
                            else 0. for c in g.candidates], dtype=float)
        targets.append((_probabilities(logits, temperature), gains / gains.sum() if gains.sum() else None))
    hard_count = sum(h is not None for _, h in targets)
    if hard_weight and not hard_count:
        raise ValueError('no retrieved positives for hard supervision')
    w = np.random.default_rng(seed).normal(0, .01, len(TEXT_FEATURE_NAMES))
    for _ in range(steps):
        grad = np.zeros_like(w)
        for x, (soft, hard), weight in zip(matrices, targets, a):
            logits = x @ w
            grad += (1 - hard_weight) * weight * temperature * (x.T @ (_probabilities(logits, temperature) - soft)) / n
            if hard is not None and hard_weight:
                grad += hard_weight * (x.T @ (_probabilities(logits, 1.) - hard)) / hard_count
        w -= learning_rate * grad
        if not np.isfinite(w).all():
            raise ValueError('training diverged')
    return {'version': 1, 'architecture': 'frozen_encoder_linear_head',
            'features': list(TEXT_FEATURE_NAMES), 'weights': w.tolist(),
            'parameter_count': len(w), 'steps': steps, 'learning_rate': learning_rate,
            'temperature': temperature, 'hard_weight': hard_weight, 'seed': seed,
            'training_queries': n, 'soft_weighting': 'mean-one per-query'}


def confidence_ablation(fit: Sequence[QueryGroup], val: Sequence[QueryGroup],
                        fit_matrices: Sequence[np.ndarray], val_matrices: Sequence[np.ndarray],
                        teacher: Mapping[tuple[str, str], float], *, steps: int = 120,
                        seed: int = 42) -> tuple[dict, dict]:
    """Four matched hybrid heads; only soft-target query weights differ."""
    from .benchmark import summarize_ranking
    from .semantic_ablation import masked_features
    train_x, test_x = masked_features(fit_matrices, 'combined'), masked_features(val_matrices, 'combined')
    metrics, predictions = {}, {}
    for scheme in ('uniform', 'margin', 'agreement', 'shuffled_margin'):
        weights = confidence_weights(fit, teacher, scheme=scheme)
        model = fit_weighted_head(fit, train_x, teacher, weights, steps=steps, seed=seed)
        scores = predict_text_head(val, test_x, model)
        predictions[scheme] = scores
        metrics[scheme] = {'metrics': summarize_ranking(val, scores, k=10),
                           'weight_min': float(weights.min()), 'weight_max': float(weights.max()),
                           'weight_std': float(weights.std()), 'effective_head_parameters': 10}
    return metrics, predictions
