"""SQLite embedding cache tests use deterministic fake encoders, never a download."""
import sqlite3

import numpy as np
import pytest

from distillranklab.scaled_text import cached_embeddings_sqlite


def fake(batch):
    return [[len(t) + 1., 2., 3.] for t in batch]


def test_new_cache_and_dedup(tmp_path):
    path = tmp_path / 'emb.sqlite'
    result, report = cached_embeddings_sqlite(['one', 'two', 'one'], model_id='test', path=path,
                                              encoder=fake, batch_size=1, log=None)
    assert set(result) == {'one', 'two'}
    assert report['new_embeddings'] == 2
    assert report['cache_hits'] == 0
    assert report['dimension'] == 3


def test_reuse_without_encoder(tmp_path):
    path = tmp_path / 'emb.sqlite'
    cached_embeddings_sqlite(['a', 'b'], model_id='test', path=path, encoder=fake, log=None)
    def forbidden(_):
        raise AssertionError('encoder should not run')
    result, report = cached_embeddings_sqlite(['b', 'a'], model_id='test', path=path,
                                              encoder=forbidden, log=None)
    assert report['cache_hits'] == 2
    assert report['new_embeddings'] == 0
    np.testing.assert_allclose(result['a'], [2, 2, 3])


def test_checkpoint_survives_failure(tmp_path):
    path = tmp_path / 'emb.sqlite'
    calls = []
    def interrupted(batch):
        calls.append(batch[0])
        if len(calls) == 2:
            raise RuntimeError('interrupted')
        return fake(batch)
    with pytest.raises(RuntimeError, match='interrupted'):
        cached_embeddings_sqlite(['a', 'b', 'c'], model_id='test', path=path,
                                 encoder=interrupted, batch_size=1, log=None)
    _, report = cached_embeddings_sqlite(['a', 'b', 'c'], model_id='test', path=path,
                                         encoder=fake, batch_size=1, log=None)
    assert report['cache_hits'] == 1
    assert report['new_embeddings'] == 2


def test_reject_model_mismatch(tmp_path):
    path = tmp_path / 'emb.sqlite'
    cached_embeddings_sqlite(['a'], model_id='one', path=path, encoder=fake, log=None)
    with pytest.raises(ValueError, match='model mismatch'):
        cached_embeddings_sqlite(['a'], model_id='two', path=path, encoder=fake, log=None)


def test_reject_dimension_mismatch(tmp_path):
    path = tmp_path / 'emb.sqlite'
    cached_embeddings_sqlite(['a'], model_id='one', path=path, encoder=fake, log=None)
    with pytest.raises(ValueError, match='invalid encoder'):
        cached_embeddings_sqlite(['b'], model_id='one', path=path,
                                 encoder=lambda _: [[1., 2., 3., 4.]], log=None)


@pytest.mark.parametrize('vectors', [[[0., 0.]], [[float('nan'), 1.]], [[1.]], [[1., 2.], [1., 2.]]])
def test_reject_invalid_encoder(tmp_path, vectors):
    with pytest.raises(ValueError, match='invalid encoder'):
        cached_embeddings_sqlite(['a'], model_id='one', path=tmp_path / 'e.sqlite',
                                 encoder=lambda _: vectors, log=None)


def test_reject_invalid_args(tmp_path):
    for kwargs in ({'batch_size': 0}, {'progress_every': 0}, {'model_id': ''}):
        with pytest.raises(ValueError):
            cached_embeddings_sqlite(['a'], path=tmp_path / 'e.sqlite',
                                     encoder=fake, log=None, **({'model_id': 'one'} | kwargs))


def test_reject_corrupt_cache(tmp_path):
    path = tmp_path / 'e.sqlite'
    cached_embeddings_sqlite(['a'], model_id='one', path=path, encoder=fake, log=None)
    with sqlite3.connect(path) as db:
        db.execute('UPDATE vectors SET vector=?', (b'garbage',))
    with pytest.raises(ValueError, match='invalid cached'):
        cached_embeddings_sqlite(['a'], model_id='one', path=path, encoder=fake, log=None)
