"""CPU-only listwise linear student for prepared SciFact candidate pools.

Training is restricted to the official training split. Unjudged candidates
are assumed nonrelevant ONLY for the hard-label objective; this is explicit.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import re
from collections.abc import Mapping, Sequence

import numpy as np

from .real_data import QueryGroup
from .ranking_metrics import evaluate_ranking

TOKEN = re.compile(r'(?u)\b\w+\b')
FEATURE_NAMES = ('token_overlap', 'query_token_recall', 'log_passage_length', 'bm25_rank_prior')
EXTRA_FEATURE_NAMES = ('saturated_term_frequency', 'query_bigram_recall',
                       'character_trigram_jaccard', 'length_ratio')
FEATURE_SETS = {
    'baseline': FEATURE_NAMES,
    'lexical': FEATURE_NAMES + EXTRA_FEATURE_NAMES[:2],
    'rich': FEATURE_NAMES + EXTRA_FEATURE_NAMES,
}


def _trigrams(value: str) -> set[str]:
    compact = ' '.join(TOKEN.findall(value.lower()))
    return {compact[i:i + 3] for i in range(max(0, len(compact) - 2))}



def feature_groups(groups: Sequence[QueryGroup], *, feature_set: str = 'baseline') -> list[np.ndarray]:
    """Deterministic, label-free features; BM25 rank is an input feature."""
    if feature_set not in FEATURE_SETS:
        raise ValueError('unknown feature set')
    matrices = []
    for group in groups:
        if not group.candidates:
            raise ValueError('query groups must have candidates')
        q = set(TOKEN.findall(group.query.lower()))
        qwords = TOKEN.findall(group.query.lower())
        qbigrams = set(zip(qwords, qwords[1:]))
        qtrigrams = _trigrams(group.query) if feature_set == 'rich' else set()
        rows = []
        for rank, candidate in enumerate(group.candidates):
            words = TOKEN.findall(candidate.passage.lower())
            counts = Counter(words)
            overlap = sum(min(counts[t], 3) for t in q)
            values = [math.log1p(overlap), len(q & set(counts)) / max(len(q), 1),
                      math.log1p(len(words)) / 10.0, 1.0 / (1.0 + rank)]
            if feature_set != 'baseline':
                values.extend((sum(counts[t] / (counts[t] + 1.0) for t in q) / max(len(q), 1),
                               len(qbigrams & set(zip(words, words[1:]))) / max(len(qbigrams), 1)))
            if feature_set == 'rich':
                dtrigrams = _trigrams(candidate.passage)
                values.extend((len(qtrigrams & dtrigrams) / max(len(qtrigrams | dtrigrams), 1),
                               min(len(words), len(qwords)) / max(len(words), len(qwords), 1)))
            rows.append(tuple(values))
        matrices.append(np.asarray(rows, dtype=np.float64))
    return matrices


def _probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / temperature
    shifted = scaled - np.max(scaled)
    values = np.exp(shifted)
    return values / values.sum()


def fit_student(groups: Sequence[QueryGroup], teacher_scores: Mapping[tuple[str, str], float] | None,
                *, mode: str = 'soft', temperature: float = 2.0, hard_weight: float = 0.5,
                steps: int = 300, learning_rate: float = 0.05, seed: int = 42,
                feature_set: str = 'baseline') -> dict:
    """Fit soft, hard or hybrid listwise objectives with full-batch gradients.

    Soft: teacher probability distribution on all candidates.
    Hard: normalized graded positive gains; unjudged treated as zero gain.
    Hybrid: convex combination of the two losses. Queries without a judged
    positive are excluded from hard loss, but included in soft loss.
    """
    if mode not in ('soft', 'hard', 'hybrid'):
        raise ValueError('mode must be soft, hard or hybrid')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be positive')
    if not math.isfinite(hard_weight) or not 0 <= hard_weight <= 1:
        raise ValueError('hard_weight must be in [0, 1]')
    if type(steps) is not int or steps < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('invalid optimizer settings')
    if type(seed) is not int:
        raise ValueError('seed must be an integer')
    if not groups or len({g.query_id for g in groups}) != len(groups):
        raise ValueError('need nonempty groups with unique query IDs')
    xs = feature_groups(groups, feature_set=feature_set)
    soft_active = mode in ('soft', 'hybrid') and (mode != 'hybrid' or hard_weight < 1)
    hard_active = mode in ('hard', 'hybrid') and (mode != 'hybrid' or hard_weight > 0)
    if soft_active and teacher_scores is None:
        raise ValueError('teacher scores required for soft training')
    soft_targets = []
    hard_targets = []
    for group in groups:
        if soft_active:
            logits = []
            for c in group.candidates:
                value = float(teacher_scores[group.query_id, c.passage_id])
                if not math.isfinite(value):
                    raise ValueError('teacher scores must be finite')
                logits.append(value)
            soft_targets.append(_probabilities(np.asarray(logits), temperature))
        else:
            soft_targets.append(None)
        gains = np.asarray([2.0 ** c.relevance - 1 if c.relevance is not None and c.relevance > 0
                            else 0.0 for c in group.candidates], dtype=np.float64)
        hard_targets.append(gains / gains.sum() if gains.sum() else None)
    if hard_active and not any(t is not None for t in hard_targets):
        raise ValueError('hard-label training requires at least one retrieved judged positive')
    w = np.random.default_rng(seed).normal(0, 0.01, size=len(FEATURE_SETS[feature_set]))
    soft_fraction = 1.0 if mode == 'soft' else 0.0 if mode == 'hard' else 1 - hard_weight
    hard_fraction = 1 - soft_fraction
    soft_count = len(groups) if soft_active else 0
    hard_count = sum(t is not None for t in hard_targets) if hard_active else 0
    for _ in range(steps):
        gradient = np.zeros_like(w)
        for x, st, ht in zip(xs, soft_targets, hard_targets):
            logits = x @ w
            if soft_active:
                gradient += soft_fraction * (x.T @ (temperature * (_probabilities(logits, temperature) - st))) / soft_count
            if hard_active and ht is not None:
                gradient += hard_fraction * (x.T @ (_probabilities(logits, 1.0) - ht)) / hard_count
        w -= learning_rate * gradient
        if not np.isfinite(w).all():
            raise ValueError('training diverged; lower learning_rate')
    return {'version': 1, 'mode': mode, 'weights': w.tolist(), 'features': list(FEATURE_SETS[feature_set]), 'feature_set': feature_set,
            'temperature': temperature, 'hard_weight': hard_weight, 'steps': steps,
            'learning_rate': learning_rate, 'seed': seed, 'training_queries': len(groups),
            'hard_eligible_queries': hard_count,
            'hard_unjudged_policy': 'nonrelevant (assumption, not verified negatives)'}


def predict_student(groups: Sequence[QueryGroup], model: dict) -> dict[tuple[str, str], float]:
    feature_set = model.get('feature_set', 'baseline')
    if (model.get('version') != 1 or feature_set not in FEATURE_SETS
            or model.get('features') != list(FEATURE_SETS[feature_set])):
        raise ValueError('incompatible student model')
    w = np.asarray(model['weights'], dtype=np.float64)
    if w.shape != (len(FEATURE_SETS[feature_set]),) or not np.isfinite(w).all():
        raise ValueError('invalid student weights')
    return {(g.query_id, c.passage_id): float(score)
            for g, x in zip(groups, feature_groups(groups, feature_set=feature_set))
            for c, score in zip(g.candidates, x @ w)}


def save_student(model: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, indent=2) + '\n', encoding='utf-8')


def evaluate_student(groups: Sequence[QueryGroup], model: dict, *, k: int = 10) -> dict:
    from .benchmark import summarize_ranking
    return summarize_ranking(groups, predict_student(groups, model), k=k)
