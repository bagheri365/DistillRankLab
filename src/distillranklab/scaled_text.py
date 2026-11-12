"""Resumable SQLite embeddings for CPU-only, frozen-encoder experiments.

A committed batch survives interruption. A model ID and embedding dimension
are checked before cached vectors are reused. No test split is accessed here.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
import hashlib
from pathlib import Path
import sqlite3
import time

import numpy as np


def cached_embeddings_sqlite(texts: Sequence[str], *, model_id: str, path: str | Path,
                             encoder: Callable[[Sequence[str]], Sequence[Sequence[float]]],
                             batch_size: int = 32, progress_every: int = 500,
                             log: Callable[[str], None] | None = print) -> tuple[dict[str, np.ndarray], dict]:
    """Encode missing unique texts in checkpointed batches, returning timing/counts.

    Only explicitly supplied ``encoder`` is called; cache hits never load a model.
    This is a single-process cache; concurrent writers are not supported.
    """
    if not model_id or type(batch_size) is not int or batch_size < 1 or type(progress_every) is not int or progress_every < 1:
        raise ValueError('invalid cache arguments')
    unique = list(dict.fromkeys(texts))
    if not unique or any(not isinstance(t, str) for t in unique):
        raise ValueError('texts must be nonempty strings')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        db.execute('CREATE TABLE IF NOT EXISTS vectors (digest TEXT PRIMARY KEY, vector BLOB NOT NULL)')
        stored = dict(db.execute('SELECT key, value FROM metadata'))
        if stored and stored.get('model_id') != model_id:
            raise ValueError('embedding cache model mismatch; use a new cache path')
        if not stored:
            db.execute('INSERT INTO metadata VALUES (?, ?)', ('model_id', model_id))
            db.commit()
        dim = int(stored['dimension']) if 'dimension' in stored else None
        digests = {t: hashlib.sha256(t.encode('utf-8')).hexdigest() for t in unique}
        found = {}
        # Point lookups avoid SQLite's bound-variable limit on large datasets.
        for t, digest in digests.items():
            row = db.execute('SELECT vector FROM vectors WHERE digest=?', (digest,)).fetchone()
            if row is not None:
                if len(row[0]) % 4:
                    raise ValueError('invalid cached embedding')
                v = np.frombuffer(row[0], dtype='<f4').copy()
                if v.size < 2 or (dim is not None and v.size != dim) or not np.isfinite(v).all() or not np.linalg.norm(v):
                    raise ValueError('invalid cached embedding')
                if dim is None:
                    dim = v.size
                found[t] = v
        missing = [t for t in unique if t not in found]
        initial_hits = len(found)
        last_report = 0
        for offset in range(0, len(missing), batch_size):
            chunk = missing[offset:offset + batch_size]
            arr = np.asarray(encoder(chunk), dtype=np.float32)
            if (arr.ndim != 2 or arr.shape[0] != len(chunk) or arr.shape[1] < 2
                    or (dim is not None and arr.shape[1] != dim)
                    or not np.isfinite(arr).all() or np.any(np.linalg.norm(arr, axis=1) == 0)):
                raise ValueError('invalid encoder embeddings')
            if dim is None:
                dim = arr.shape[1]
                db.execute('INSERT OR REPLACE INTO metadata VALUES (?, ?)', ('dimension', str(dim)))
            for t, v in zip(chunk, arr):
                db.execute('INSERT OR REPLACE INTO vectors VALUES (?, ?)',
                           (digests[t], sqlite3.Binary(np.asarray(v, dtype='<f4').tobytes())))
                found[t] = v.copy()
            db.commit()  # Durable checkpoint after each batch.
            completed = offset + len(chunk)
            if log is not None and (completed - last_report >= progress_every or completed == len(missing)):
                log(f'embeddings: {completed}/{len(missing)} new texts cached ({initial_hits} reused)')
                last_report = completed
    return found, {'unique_texts': len(unique), 'cache_hits': initial_hits,
                   'new_embeddings': len(missing), 'dimension': dim,
                   'seconds': time.perf_counter() - started, 'cache': str(path)}
