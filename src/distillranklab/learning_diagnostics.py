"""Training-trajectory and teacher-agreement diagnostics for fixed-pool heads.

Exploratory only: fit-query teacher agreement is not held-out evidence.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import numpy as np

from .real_data import QueryGroup
from .semantic_ablation import masked_features
from .text_student import TEXT_FEATURE_NAMES, fit_text_head, predict_text_head

SUPERVISION = {'teacher_only': 0.0, 'hybrid': 0.5, 'relevance_only': 1.0}


def ordering_diagnostics(groups: Sequence[QueryGroup], baseline: Mapping,
                         challenger: Mapping) -> dict:
    """Count exact candidate-order changes, ties broken by original pool order."""
    if not groups:
        raise ValueError('empty groups')
    changed_top1 = changed_top10 = changed_full = 0
    total_displaced = 0
    for group in groups:
        keys = [(group.query_id, c.passage_id) for c in group.candidates]
        a = np.asarray([baseline[k] for k in keys], dtype=float)
        b = np.asarray([challenger[k] for k in keys], dtype=float)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('nonfinite scores')
        order_a = np.argsort(-a, kind='stable')
        order_b = np.argsort(-b, kind='stable')
        changed_top1 += int(order_a[0] != order_b[0])
        changed_top10 += int(not np.array_equal(order_a[:10], order_b[:10]))
        changed_full += int(not np.array_equal(order_a, order_b))
        total_displaced += len(set(order_a[:10]) ^ set(order_b[:10])) // 2
    return {'queries': len(groups), 'changed_top1': changed_top1,
            'changed_top10_order': changed_top10, 'changed_full_order': changed_full,
            'mean_top10_membership_replacements': total_displaced / len(groups)}


def teacher_agreement(groups: Sequence[QueryGroup], teacher: Mapping,
                      student: Mapping) -> dict:
    """Fit-query teacher agreement: top-1 and per-query Spearman rank rho.

    Spearman ranks use stable candidate-pool order for tied scores; this is a
    descriptive measure and not a relevance-based ranking metric.
    """
    if not groups:
        raise ValueError('empty groups')
    top1 = 0
    rho = []
    for group in groups:
        keys = [(group.query_id, c.passage_id) for c in group.candidates]
        t = np.asarray([teacher[k] for k in keys], dtype=float)
        s = np.asarray([student[k] for k in keys], dtype=float)
        if not np.isfinite(t).all() or not np.isfinite(s).all():
            raise ValueError('nonfinite scores')
        ot = np.argsort(-t, kind='stable')
        os = np.argsort(-s, kind='stable')
        top1 += int(ot[0] == os[0])
        if len(keys) > 1:
            rt = np.empty(len(keys), dtype=float)
            rs = np.empty(len(keys), dtype=float)
            rt[ot] = np.arange(len(keys))
            rs[os] = np.arange(len(keys))
            rho.append(float(1 - 6 * np.sum((rt - rs) ** 2) / (len(keys) * (len(keys) ** 2 - 1))))
    return {'fit_queries': len(groups), 'teacher_top1_agreement': top1 / len(groups),
            'mean_spearman_rho': float(np.mean(rho)) if rho else None,
            'ties': 'stable pool-order tie breaking'}


def learning_diagnostics(fit: Sequence[QueryGroup], validation: Sequence[QueryGroup],
                         fit_matrices: Sequence[np.ndarray], validation_matrices: Sequence[np.ndarray],
                         teacher: Mapping, *, steps: int = 120, seed: int = 42) -> dict:
    """Replay six matched heads with optimizer checkpoints and ranking checks."""
    if not fit or not validation or steps < 1:
        raise ValueError('empty groups or invalid steps')
    result = {}
    for block in ('semantic_only', 'combined'):
        train_x = masked_features(fit_matrices, block)
        val_x = masked_features(validation_matrices, block)
        init_w = np.random.default_rng(seed).normal(0, .01, len(TEXT_FEATURE_NAMES))
        initial = {'version': 1, 'features': list(TEXT_FEATURE_NAMES), 'weights': init_w.tolist()}
        initial_val = predict_text_head(validation, val_x, initial)
        initial_fit = predict_text_head(fit, train_x, initial)
        models = {}
        for objective, hard_weight in SUPERVISION.items():
            trace = []
            model = fit_text_head(fit, train_x, teacher, steps=steps, learning_rate=.05,
                                  temperature=2., hard_weight=hard_weight, seed=seed, trace=trace)
            val_scores = predict_text_head(validation, val_x, model)
            fit_scores = predict_text_head(fit, train_x, model)
            models[objective] = {'trajectory': trace,
                                 'active_weights': {TEXT_FEATURE_NAMES[i]: model['weights'][i]
                                                    for i in ((8, 9) if block == 'semantic_only' else range(10))},
                                 'validation_order_vs_initial': ordering_diagnostics(validation, initial_val, val_scores),
                                 'fit_teacher_agreement': teacher_agreement(fit, teacher, fit_scores),
                                 'fit_teacher_agreement_initial': teacher_agreement(fit, teacher, initial_fit)}
        result[block] = models
    return result
