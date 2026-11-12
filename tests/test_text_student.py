import json

import numpy as np
import pytest

from distillranklab.real_data import Candidate, QueryGroup
from distillranklab.text_student import (TEXT_FEATURE_NAMES, encode_cached, fit_text_head,
                                         predict_text_head, text_feature_groups)


def groups():
    return (QueryGroup('q1', 'red planet', (Candidate('p1', 'mars red planet', 1),
                                           Candidate('p2', 'ocean blue water', None))),
            QueryGroup('q2', 'earth ocean', (Candidate('p3', 'ocean on earth', 1),
                                            Candidate('p4', 'distant moon', 0))))


def fake_encoder(texts):
    return [[1 + len(t) % 5, 1 + sum(map(ord, t)) % 7, 2] for t in texts]


def test_cache_reuse_and_model_binding(tmp_path):
    path = tmp_path / 'embed.json'
    values = encode_cached(['a', 'b', 'a'], encoder=fake_encoder, model_id='fake', cache_path=path)
    assert len(values) == 2
    assert json.loads(path.read_text())['model_id'] == 'fake'
    assert np.allclose(encode_cached(['b', 'a'], encoder=lambda _: (_ for _ in ()).throw(AssertionError()),
                                     model_id='fake', cache_path=path)['a'], values['a'])
    with pytest.raises(ValueError, match='mismatch'):
        encode_cached(['a'], encoder=fake_encoder, model_id='other', cache_path=path)


def test_missing_cache_entries_require_encoder(tmp_path):
    path = tmp_path / 'emb.json'
    encode_cached(['a'], encoder=fake_encoder, model_id='fake', cache_path=path)
    with pytest.raises(RuntimeError):
        encode_cached(['a', 'b'], encoder=lambda _: (_ for _ in ()).throw(RuntimeError()),
                      model_id='fake', cache_path=path)
    assert len(json.loads(path.read_text())['vectors']) == 1


@pytest.mark.parametrize('encoder', [lambda _: [[1]], lambda _: [[float('nan'), 2]],
                                      lambda _: [[0, 0]], lambda _: [[1, 2], [2, 3]]])
def test_invalid_encoder_rejected(tmp_path, encoder):
    with pytest.raises(ValueError):
        encode_cached(['a'], encoder=encoder, model_id='fake', cache_path=tmp_path / 'e.json')


def test_train_predict_and_no_leakage(tmp_path):
    gs = groups()
    texts = [t for g in gs for t in [g.query, *(c.passage for c in g.candidates)]]
    emb = encode_cached(texts, encoder=fake_encoder, model_id='fake', cache_path=tmp_path / 'emb.json')
    xs = text_feature_groups(gs, emb)
    assert xs[0].shape == (2, len(TEXT_FEATURE_NAMES))
    teacher = {(g.query_id, c.passage_id): float(c.relevance or 0) for g in gs for c in g.candidates}
    model = fit_text_head(gs, xs, teacher, steps=10)
    assert model['parameter_count'] == 10
    scores = predict_text_head(gs, xs, model)
    assert len(scores) == 4 and all(np.isfinite(list(scores.values())))
    assert fit_text_head(gs, xs, teacher, steps=10) == model
    with pytest.raises(ValueError):
        predict_text_head(gs, xs, {**model, 'weights': [0]})


def test_no_positive_hard_rejected():
    gs = (QueryGroup('q', 'hello', (Candidate('p', 'world', None),)),)
    xs = [np.zeros((1, 10))]
    with pytest.raises(ValueError, match='positives'):
        fit_text_head(gs, xs, {('q', 'p'): 0.0}, hard_weight=0.5)
    assert fit_text_head(gs, xs, {('q', 'p'): 0.0}, hard_weight=0, steps=1)['training_queries'] == 1


def test_invalid_feature_matrix():
    gs = groups()
    teacher = {(g.query_id, c.passage_id): 0. for g in gs for c in g.candidates}
    with pytest.raises(ValueError):
        fit_text_head(gs, [np.zeros((2, 8)), np.zeros((2, 10))], teacher)
