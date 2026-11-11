"""Offline SciFact archive and candidate-generation tests."""
import json
import zipfile

import pytest

from distillranklab.real_data import load_groups
from distillranklab.scifact import prepare_scifact, retrieve_bm25


def archive_fixture(path):
    with zipfile.ZipFile(path, 'w') as zf:
        zf.writestr('scifact/corpus.jsonl', '\n'.join(json.dumps(row) for row in [
            {'_id': 'a', 'title': 'Apples', 'text': 'Apples are red fruit'},
            {'_id': 'b', 'title': 'Bananas', 'text': 'Bananas are yellow fruit'},
            {'_id': 'c', 'title': 'Cells', 'text': 'Cells have membranes'}]))
        zf.writestr('scifact/queries.jsonl', '\n'.join(json.dumps(row) for row in [
            {'_id': 'q1', 'text': 'red apples'}, {'_id': 'q2', 'text': 'yellow bananas'},
            {'_id': 'q3', 'text': 'cell membrane'}]))
        zf.writestr('scifact/qrels/train.tsv', 'query-id\tcorpus-id\tscore\nq3\tc\t1\n')
        zf.writestr('scifact/qrels/test.tsv', 'query-id\tcorpus-id\tscore\nq1\ta\t1\nq2\tb\t1\n')


def test_bm25_ranking_and_ties():
    corpus = {'b': 'cat', 'a': 'cat', 'c': 'dog'}
    assert retrieve_bm25(corpus, {'q': 'cat'}, top_k=3)['q'] == ('a', 'b', 'c')
    assert retrieve_bm25(corpus, {'q': 'unseen'}, top_k=3)['q'] == ('a', 'b', 'c')


def test_candidate_generation_independent_of_qrels(tmp_path):
    archive = tmp_path / 'scifact.zip'
    archive_fixture(archive)
    first_dir = tmp_path / 'first'
    prepare_scifact(archive, first_dir, top_k=2)
    with zipfile.ZipFile(archive) as zf:
        original = {name: zf.read(name) for name in zf.namelist()}
    original['scifact/qrels/test.tsv'] = b'query-id\tcorpus-id\tscore\nq1\tc\t1\nq2\ta\t1\n'
    with zipfile.ZipFile(archive, 'w') as zf:
        for name, contents in original.items():
            zf.writestr(name, contents)
    second_dir = tmp_path / 'second'
    prepare_scifact(archive, second_dir, top_k=2)
    assert (first_dir / 'candidates.tsv').read_bytes() == (second_dir / 'candidates.tsv').read_bytes()
    assert (first_dir / 'qrels.tsv').read_bytes() != (second_dir / 'qrels.tsv').read_bytes()


def test_export_and_split(tmp_path):
    archive = tmp_path / 'scifact.zip'
    archive_fixture(archive)
    test_dir = tmp_path / 'test'
    train_dir = tmp_path / 'train'
    manifest = prepare_scifact(archive, test_dir, top_k=2)
    prepare_scifact(archive, train_dir, split='train', top_k=2)
    def groups(path):
        return load_groups(*(path / f'{name}.tsv' for name in ('queries', 'passages', 'candidates', 'qrels')))
    test = groups(test_dir)
    train = groups(train_dir)
    assert {g.query_id for g in test} == {'q1', 'q2'}
    assert {g.query_id for g in train} == {'q3'}
    assert manifest['positive_injection'] is False
    assert manifest['queries_with_retrieved_positive'] == 2
    assert all(len(g.candidates) == 2 for g in test)
    assert json.loads((test_dir / 'manifest.json').read_text()) == manifest


def test_validation(tmp_path):
    archive = tmp_path / 'scifact.zip'
    archive_fixture(archive)
    with pytest.raises(ValueError, match='top_k'):
        prepare_scifact(archive, tmp_path / 'out', top_k=0)
    with pytest.raises(ValueError, match='max_queries'):
        prepare_scifact(archive, tmp_path / 'out', max_queries=0)
    with pytest.raises(ValueError, match='split'):
        prepare_scifact(archive, tmp_path / 'out', split='dev')
