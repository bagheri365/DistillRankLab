"""Batched, cached cross-encoder scoring for query/passage groups.

The optional sentence-transformers dependency is imported only for real inference.
Cache entries are keyed by model identifier and exact input texts, not just IDs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Callable, Sequence

from .real_data import QueryGroup

DEFAULT_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
Pair = tuple[str, str]
Scorer = Callable[[Sequence[Pair]], Sequence[float]]


def cross_encoder_scorer(model_name: str = DEFAULT_MODEL, *, device: str = 'cpu') -> Scorer:
    """Load a frozen cross-encoder lazily; requires sentence-transformers.

    The model is used for inference only, and never fine-tuned here.
    """
    if not model_name or not isinstance(model_name, str):
        raise ValueError('model_name must be a nonempty string')
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise ImportError('Install optional inference dependencies: pip install sentence-transformers') from exc
    model = CrossEncoder(model_name, device=device)

    def predict(pairs: Sequence[Pair]) -> Sequence[float]:
        return model.predict(list(pairs), show_progress_bar=False).tolist()

    return predict


def _key(model_id: str, query: str, passage: str) -> str:
    payload = json.dumps([model_id, query, passage], ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _load_cache(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f'invalid teacher cache: {path}') from exc
    if not isinstance(obj, dict) or obj.get('version') != 1 or not isinstance(obj.get('scores'), dict):
        raise ValueError(f'invalid teacher cache format: {path}')
    scores = obj['scores']
    if any(not isinstance(k, str) or not isinstance(v, (int, float)) or isinstance(v, bool)
           or not math.isfinite(v) for k, v in scores.items()):
        raise ValueError(f'invalid teacher cache scores: {path}')
    return {k: float(v) for k, v in scores.items()}


def _save_cache(path: Path, scores: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.teacher-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump({'version': 1, 'scores': scores}, handle, sort_keys=True)
            handle.write('\n')
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def score_groups(groups: Sequence[QueryGroup], *, scorer: Scorer,
                 model_id: str, cache_path: str | Path, batch_size: int = 16) -> dict[tuple[str, str], float]:
    """Return scores keyed by (query_id, passage_id), caching exact text pairs.

    All candidate pairs are scored, including unjudged passages. Duplicate
    query/passage IDs are rejected. Cache updates are atomic on success.
    """
    if not isinstance(model_id, str) or not model_id:
        raise ValueError('model_id must be nonempty')
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError('batch_size must be a positive integer')
    if not callable(scorer):
        raise ValueError('scorer must be callable')
    cache = _load_cache(Path(cache_path))
    entries: list[tuple[tuple[str, str], str, Pair]] = []
    seen: set[tuple[str, str]] = set()
    for group in groups:
        for candidate in group.candidates:
            identifier = (group.query_id, candidate.passage_id)
            if identifier in seen:
                raise ValueError('duplicate query/passage identifier')
            seen.add(identifier)
            pair = (group.query, candidate.passage)
            entries.append((identifier, _key(model_id, *pair), pair))
    missing: dict[str, Pair] = {}
    for _, key, pair in entries:
        if key not in cache:
            missing[key] = pair
    items = list(missing.items())
    for offset in range(0, len(items), batch_size):
        batch = items[offset:offset + batch_size]
        predictions = list(scorer([pair for _, pair in batch]))
        if len(predictions) != len(batch):
            raise ValueError('scorer returned incorrect number of scores')
        for (key, _), value in zip(batch, predictions):
            if isinstance(value, bool):
                raise ValueError('teacher scores must be finite real numbers')
            try:
                score = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError('teacher scores must be finite real numbers') from exc
            if not math.isfinite(score):
                raise ValueError('teacher scores must be finite real numbers')
            cache[key] = score
    if items:
        _save_cache(Path(cache_path), cache)
    return {identifier: cache[key] for identifier, key, _ in entries}
