"""CPU frozen text-encoder features and trainable listwise linear ranking head.

The encoder is never fine-tuned. Optional sentence-transformers dependency is
loaded only by the CLI; tests inject a deterministic fake encoder.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .real_data import QueryGroup
from .real_student import FEATURE_SETS, _probabilities, feature_groups

TEXT_FEATURE_NAMES = tuple(FEATURE_SETS['rich']) + ('embedding_cosine', 'embedding_max_product')
Encoder = Callable[[Sequence[str]], Sequence[Sequence[float]]]


def frozen_sentence_encoder(model_id: str, *, batch_size: int = 16) -> Encoder:
    """Explicit opt-in to downloading/loading a pretrained encoder on CPU."""
    if not model_id or batch_size < 1:
        raise ValueError('invalid encoder configuration')
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError('install sentence-transformers to enable text embeddings') from exc
    model = SentenceTransformer(model_id, device='cpu')

    def encode(texts: Sequence[str]) -> np.ndarray:
        return np.asarray(model.encode(list(texts), batch_size=batch_size,
                                       convert_to_numpy=True, show_progress_bar=False), dtype=float)
    return encode


def _key(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def encode_cached(texts: Sequence[str], *, encoder: Encoder, model_id: str,
                  cache_path: str | Path, batch_size: int = 32) -> dict[str, np.ndarray]:
    """Validate model-bound cache and encode only missing unique texts."""
    if not model_id or type(batch_size) is not int or batch_size < 1:
        raise ValueError('invalid cache settings')
    path = Path(cache_path)
    if path.exists():
        cache = json.loads(path.read_text(encoding='utf-8'))
        if cache.get('version') != 1 or cache.get('model_id') != model_id or not isinstance(cache.get('vectors'), dict):
            raise ValueError('embedding cache model/version mismatch')
    else:
        cache = {'version': 1, 'model_id': model_id, 'vectors': {}}
    unique = dict.fromkeys(texts)
    vectors = cache['vectors']
    missing = [t for t in unique if _key(t) not in vectors]
    for offset in range(0, len(missing), batch_size):
        chunk = missing[offset:offset + batch_size]
        arr = np.asarray(encoder(chunk), dtype=float)
        if arr.ndim != 2 or arr.shape[0] != len(chunk) or arr.shape[1] < 2 or not np.isfinite(arr).all():
            raise ValueError('encoder returned invalid embeddings')
        for t, v in zip(chunk, arr):
            vectors[_key(t)] = v.tolist()
    result = {t: np.asarray(vectors[_key(t)], dtype=float) for t in unique}
    dims = {v.shape for v in result.values()}
    if len(dims) != 1 or any(v.ndim != 1 or not np.isfinite(v).all() or np.linalg.norm(v) == 0
                             for v in result.values()):
        raise ValueError('invalid or inconsistent cached embeddings')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, separators=(',', ':')) + '\n', encoding='utf-8')
    return result


def text_feature_groups(groups: Sequence[QueryGroup], embeddings: Mapping[str, np.ndarray]) -> list[np.ndarray]:
    lexical = feature_groups(groups, feature_set='rich')
    matrices = []
    for group, base in zip(groups, lexical):
        q = np.asarray(embeddings[group.query], dtype=float)
        rows = []
        for candidate in group.candidates:
            d = np.asarray(embeddings[candidate.passage], dtype=float)
            if q.ndim != 1 or q.shape != d.shape or not np.isfinite(q).all() or not np.isfinite(d).all():
                raise ValueError('incompatible text embeddings')
            norm = np.linalg.norm(q) * np.linalg.norm(d)
            if norm <= 0:
                raise ValueError('zero embedding norm')
            product = (q / np.linalg.norm(q)) * (d / np.linalg.norm(d))
            rows.append((float(product.sum()), float(product.max())))
        matrices.append(np.column_stack((base, np.asarray(rows, dtype=float))))
    return matrices


def fit_text_head(groups: Sequence[QueryGroup], matrices: Sequence[np.ndarray],
                  teacher_scores: Mapping[tuple[str, str], float], *, steps: int = 120,
                  learning_rate: float = 0.05, temperature: float = 2.0,
                  hard_weight: float = 0.5, seed: int = 42,
                  trace: list[dict] | None = None) -> dict:
    """Hybrid listwise distillation; hard supervision excludes no-positive queries."""
    if (not groups or len(groups) != len(matrices) or len({g.query_id for g in groups}) != len(groups)
            or type(steps) is not int or steps < 1 or not math.isfinite(learning_rate)
            or learning_rate <= 0 or not math.isfinite(temperature) or temperature <= 0
            or not math.isfinite(hard_weight) or not 0 <= hard_weight <= 1):
        raise ValueError('invalid head training arguments')
    targets = []
    for g, x in zip(groups, matrices):
        if x.shape != (len(g.candidates), len(TEXT_FEATURE_NAMES)) or not np.isfinite(x).all():
            raise ValueError('invalid feature matrix')
        logits = np.asarray([teacher_scores[g.query_id, c.passage_id] for c in g.candidates], dtype=float)
        if not np.isfinite(logits).all():
            raise ValueError('invalid teacher scores')
        gains = np.asarray([2.0 ** c.relevance - 1 if c.relevance is not None and c.relevance > 0
                            else 0.0 for c in g.candidates], dtype=float)
        targets.append((_probabilities(logits, temperature), gains / gains.sum() if gains.sum() else None))
    hard_count = sum(h is not None for _, h in targets)
    if hard_weight and not hard_count:
        raise ValueError('hard supervision requires retrieved positives')
    w = np.random.default_rng(seed).normal(0, 0.01, len(TEXT_FEATURE_NAMES))
    def record(step: int) -> None:
        if trace is None:
            return
        soft_kl = 0.0
        hard_ce = 0.0
        for x, (soft, hard) in zip(matrices, targets):
            pred_soft = _probabilities(x @ w, temperature)
            soft_kl += float(np.sum(soft * np.log(np.maximum(soft, 1e-300) /
                                                    np.maximum(pred_soft, 1e-300)))) / len(groups)
            if hard is not None:
                pred_hard = _probabilities(x @ w, 1.0)
                hard_ce -= float(np.sum(hard * np.log(np.maximum(pred_hard, 1e-300)))) / hard_count
        trace.append({'step': step, 'soft_kl': soft_kl, 'hard_cross_entropy': hard_ce,
                      'weight_norm': float(np.linalg.norm(w)), 'weights': w.tolist()})

    record(0)
    checkpoints = {1, 10, 30, 60, steps}
    for step in range(1, steps + 1):
        grad = np.zeros_like(w)
        for x, (soft, hard) in zip(matrices, targets):
            logits = x @ w
            grad += (1 - hard_weight) * temperature * (x.T @ (_probabilities(logits, temperature) - soft)) / len(groups)
            if hard is not None and hard_weight:
                grad += hard_weight * (x.T @ (_probabilities(logits, 1) - hard)) / hard_count
        w -= learning_rate * grad
        if not np.isfinite(w).all():
            raise ValueError('training diverged')
        if trace is not None and step in checkpoints:
            record(step)
    return {'version': 1, 'architecture': 'frozen_encoder_linear_head', 'features': list(TEXT_FEATURE_NAMES),
            'weights': w.tolist(), 'parameter_count': len(w), 'steps': steps,
            'learning_rate': learning_rate, 'temperature': temperature,
            'hard_weight': hard_weight, 'seed': seed, 'training_queries': len(groups),
            'hard_unjudged_policy': 'nonrelevant (assumption, not verified negatives)'}


def predict_text_head(groups: Sequence[QueryGroup], matrices: Sequence[np.ndarray], model: dict) -> dict:
    if model.get('version') != 1 or model.get('features') != list(TEXT_FEATURE_NAMES):
        raise ValueError('incompatible text head')
    w = np.asarray(model['weights'], dtype=float)
    if w.shape != (len(TEXT_FEATURE_NAMES),) or not np.isfinite(w).all() or len(groups) != len(matrices):
        raise ValueError('invalid text head weights or groups')
    result = {}
    for g, x in zip(groups, matrices):
        if x.shape != (len(g.candidates), len(w)):
            raise ValueError('invalid feature shape')
        for c, value in zip(g.candidates, x @ w):
            result[g.query_id, c.passage_id] = float(value)
    return result
