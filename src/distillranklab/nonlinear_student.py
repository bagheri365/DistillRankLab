"""Small NumPy tanh-MLP listwise ranker; train-only teacher supervision."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np

from .real_data import QueryGroup
from .real_student import FEATURE_SETS, _probabilities, feature_groups


def parameter_count(input_dim: int, hidden_units: int) -> int:
    if type(input_dim) is not int or input_dim < 1 or type(hidden_units) is not int or hidden_units < 1:
        raise ValueError('positive dimensions required')
    return input_dim * hidden_units + hidden_units + hidden_units + 1


def fit_nonlinear(groups: Sequence[QueryGroup],
                  teacher_scores: Mapping[tuple[str, str], float] | None, *,
                  mode: str = 'soft', feature_set: str = 'rich', hidden_units: int = 8,
                  temperature: float = 2.0, hard_weight: float = 0.5,
                  steps: int = 120, learning_rate: float = 0.05,
                  weight_decay: float = 0.0, seed: int = 42) -> dict:
    """Full-batch query-averaged listwise gradients; no test data access.

    Unjudged documents are assumed zero gain for hard supervision. Queries
    without retrieved positives contribute to soft loss, not hard loss.
    """
    if mode not in ('hard', 'soft', 'hybrid') or feature_set not in FEATURE_SETS:
        raise ValueError('invalid mode or feature set')
    if (type(hidden_units) is not int or hidden_units < 1 or type(steps) is not int or steps < 1
            or type(seed) is not int or not math.isfinite(temperature) or temperature <= 0
            or not math.isfinite(learning_rate) or learning_rate <= 0
            or not math.isfinite(weight_decay) or weight_decay < 0
            or not math.isfinite(hard_weight) or not 0 <= hard_weight <= 1):
        raise ValueError('invalid optimizer settings')
    if not groups or len({g.query_id for g in groups}) != len(groups):
        raise ValueError('need unique nonempty query groups')
    xs = feature_groups(groups, feature_set=feature_set)
    soft_fraction = 1.0 if mode == 'soft' else 0.0 if mode == 'hard' else 1 - hard_weight
    hard_fraction = 1 - soft_fraction
    if soft_fraction and teacher_scores is None:
        raise ValueError('teacher scores required for soft training')
    soft_targets, hard_targets = [], []
    for group in groups:
        if soft_fraction:
            logits = np.array([float(teacher_scores[grouP]) for grouP in
                               ((group.query_id, c.passage_id) for c in group.candidates)])
            if not np.isfinite(logits).all():
                raise ValueError('teacher scores must be finite')
            soft_targets.append(_probabilities(logits, temperature))
        else:
            soft_targets.append(None)
        gains = np.array([2.0 ** c.relevance - 1 if c.relevance is not None and c.relevance > 0
                          else 0.0 for c in group.candidates], dtype=float)
        hard_targets.append(gains / gains.sum() if gains.sum() else None)
    hard_count = sum(t is not None for t in hard_targets) if hard_fraction else 0
    if hard_fraction and not hard_count:
        raise ValueError('hard supervision requires retrieved positives')
    rng = np.random.default_rng(seed)
    dim = len(FEATURE_SETS[feature_set])
    w1 = rng.normal(0, 0.1 / math.sqrt(dim), (dim, hidden_units))
    b1 = np.zeros(hidden_units)
    w2 = rng.normal(0, 0.1 / math.sqrt(hidden_units), hidden_units)
    b2 = 0.0
    for _ in range(steps):
        gw1 = np.zeros_like(w1)
        gb1 = np.zeros_like(b1)
        gw2 = np.zeros_like(w2)
        gb2 = 0.0
        for x, soft, hard in zip(xs, soft_targets, hard_targets):
            hidden = np.tanh(x @ w1 + b1)
            logits = hidden @ w2 + b2
            grad = np.zeros(len(x))
            if soft_fraction:
                grad += soft_fraction * temperature * (_probabilities(logits, temperature) - soft) / len(xs)
            if hard_fraction and hard is not None:
                grad += hard_fraction * (_probabilities(logits, 1.0) - hard) / hard_count
            gw2 += hidden.T @ grad
            gb2 += float(grad.sum())
            back = (grad[:, None] * w2[None, :]) * (1 - hidden ** 2)
            gw1 += x.T @ back
            gb1 += back.sum(axis=0)
        w1 -= learning_rate * (gw1 + weight_decay * w1)
        b1 -= learning_rate * gb1
        w2 -= learning_rate * (gw2 + weight_decay * w2)
        b2 -= learning_rate * gb2
        if not (np.isfinite(w1).all() and np.isfinite(w2).all()
                and np.isfinite(b1).all() and math.isfinite(b2)):
            raise ValueError('training diverged')
    return {'version': 1, 'architecture': 'tanh_mlp', 'feature_set': feature_set,
            'features': list(FEATURE_SETS[feature_set]), 'hidden_units': hidden_units,
            'w1': w1.tolist(), 'b1': b1.tolist(), 'w2': w2.tolist(), 'b2': b2,
            'mode': mode, 'temperature': temperature, 'hard_weight': hard_weight,
            'learning_rate': learning_rate, 'steps': steps, 'weight_decay': weight_decay,
            'seed': seed, 'training_queries': len(groups), 'hard_eligible_queries': hard_count,
            'parameter_count': parameter_count(dim, hidden_units),
            'hard_unjudged_policy': 'nonrelevant (assumption, not verified negatives)'}


def predict_nonlinear(groups: Sequence[QueryGroup], model: dict) -> dict[tuple[str, str], float]:
    feature_set = model.get('feature_set')
    hidden = model.get('hidden_units')
    if (model.get('version') != 1 or model.get('architecture') != 'tanh_mlp'
            or feature_set not in FEATURE_SETS or type(hidden) is not int or hidden < 1
            or model.get('features') != list(FEATURE_SETS[feature_set])):
        raise ValueError('incompatible nonlinear student')
    dim = len(FEATURE_SETS[feature_set])
    w1, b1 = np.asarray(model['w1'], dtype=float), np.asarray(model['b1'], dtype=float)
    w2, b2 = np.asarray(model['w2'], dtype=float), float(model['b2'])
    if (w1.shape != (dim, hidden) or b1.shape != (hidden,) or w2.shape != (hidden,)
            or not all(np.isfinite(v).all() for v in (w1, b1, w2, np.asarray(b2)))):
        raise ValueError('invalid nonlinear student parameters')
    return {(g.query_id, c.passage_id): float(score)
            for g, x in zip(groups, feature_groups(groups, feature_set=feature_set))
            for c, score in zip(g.candidates, np.tanh(x @ w1 + b1) @ w2 + b2)}
