"""Tests do not download a neural model or require sentence-transformers."""

import json
from pathlib import Path

import pytest

from distillranklab.real_data import Candidate, QueryGroup, load_groups
from distillranklab.teacher import score_groups


FIXTURE = Path(__file__).parent / 'fixtures' / 'tiny_ranking'


def groups():
    return load_groups(*(FIXTURE / name for name in
                         ('queries.tsv', 'passages.tsv', 'candidates.tsv', 'qrels.tsv')))


def test_scores_all_candidates_including_unjudged(tmp_path):
    seen = []

    def scorer(pairs):
        seen.extend(pairs)
        return [float(i) for i, _ in enumerate(pairs)]

    result = score_groups(groups(), scorer=scorer, model_id='mock',
                          cache_path=tmp_path / 'cache.json', batch_size=3)
    assert len(result) == 10
    assert len(seen) == 10
    assert (groups()[1].query_id, groups()[1].candidates[1].passage_id) in result


def test_cache_hit_skips_inference(tmp_path):
    path = tmp_path / 'cache.json'
    first = score_groups(groups(), scorer=lambda pairs: [1.25] * len(pairs),
                         model_id='mock', cache_path=path)

    def fail(_):
        raise AssertionError('cache should be used')

    assert score_groups(groups(), scorer=fail, model_id='mock', cache_path=path) == first


def test_model_change_invalidates_cache(tmp_path):
    path = tmp_path / 'cache.json'
    a = score_groups(groups(), scorer=lambda pairs: [1] * len(pairs),
                     model_id='a', cache_path=path)
    b = score_groups(groups(), scorer=lambda pairs: [2] * len(pairs),
                     model_id='b', cache_path=path)
    assert all(v == 1 for v in a.values())
    assert all(v == 2 for v in b.values())


def test_text_change_invalidates_only_changed_pair(tmp_path):
    path = tmp_path / 'cache.json'
    original = groups()
    score_groups(original, scorer=lambda pairs: [1] * len(pairs),
                 model_id='mock', cache_path=path)
    changed = (QueryGroup(original[0].query_id, 'modified query', original[0].candidates),) + original[1:]
    calls = []

    def scorer(pairs):
        calls.extend(pairs)
        return [2] * len(pairs)

    result = score_groups(changed, scorer=scorer, model_id='mock', cache_path=path)
    assert len(calls) == len(original[0].candidates)
    assert result[(original[0].query_id, original[0].candidates[0].passage_id)] == 2


def test_batch_sizes(tmp_path):
    sizes = []

    def scorer(pairs):
        sizes.append(len(pairs))
        return [0] * len(pairs)

    score_groups(groups(), scorer=scorer, model_id='mock',
                 cache_path=tmp_path / 'cache.json', batch_size=4)
    assert sizes == [4, 4, 2]


@pytest.mark.parametrize('scores', [[float('nan')], [float('inf')], ['bad'], []])
def test_invalid_predictions_rejected(tmp_path, scores):
    with pytest.raises(ValueError):
        score_groups(groups()[:1], scorer=lambda pairs: scores,
                     model_id='mock', cache_path=tmp_path / 'cache.json')
    assert not (tmp_path / 'cache.json').exists()


def test_duplicate_ids_rejected(tmp_path):
    repeated = groups()[:1] * 2
    with pytest.raises(ValueError, match='duplicate'):
        score_groups(repeated, scorer=lambda pairs: [0] * len(pairs),
                     model_id='mock', cache_path=tmp_path / 'cache.json')


@pytest.mark.parametrize('kwargs', [{'model_id': ''}, {'batch_size': 0}, {'batch_size': 1.5}, {'scorer': None}])
def test_invalid_parameters(tmp_path, kwargs):
    params = dict(scorer=lambda pairs: [0] * len(pairs), model_id='mock',
                  cache_path=tmp_path / 'cache.json')
    params.update(kwargs)
    with pytest.raises(ValueError):
        score_groups(groups(), **params)


def test_invalid_cache_rejected(tmp_path):
    path = tmp_path / 'cache.json'
    path.write_text('{"version": 1, "scores": {"x": "oops"}}', encoding='utf-8')
    with pytest.raises(ValueError, match='cache'):
        score_groups(groups(), scorer=lambda pairs: [0] * len(pairs),
                     model_id='mock', cache_path=path)


def test_cache_file_schema(tmp_path):
    path = tmp_path / 'cache.json'
    score_groups(groups(), scorer=lambda pairs: [0] * len(pairs),
                 model_id='mock', cache_path=path)
    saved = json.loads(path.read_text())
    assert saved['version'] == 1
    assert len(saved['scores']) == 10


def test_empty_groups_do_not_write_cache(tmp_path):
    path = tmp_path / 'cache.json'
    assert score_groups((), scorer=lambda pairs: [], model_id='mock', cache_path=path) == {}
    assert not path.exists()
