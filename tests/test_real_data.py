"""Real-data TSV loader and query-disjoint split regression tests."""
from pathlib import Path

import pytest

from distillranklab.real_data import judgment_coverage, load_groups, split_groups


BASE = Path(__file__).parent / 'fixtures' / 'tiny_ranking'


def load_fixture():
    return load_groups(*(BASE / name for name in (
        'queries.tsv', 'passages.tsv', 'candidates.tsv', 'qrels.tsv')))


def test_fixture_counts_and_judgment_status():
    groups = load_fixture()
    assert len(groups) == 5
    assert judgment_coverage(groups) == dict(
        candidates=10, judged_relevant=5, judged_nonrelevant=1, unjudged=4)
    assert groups[0].candidates[0].relevance == 1
    assert groups[0].candidates[1].relevance == 0
    assert groups[1].candidates[1].relevance is None


def test_query_disjoint_and_deterministic():
    groups = load_fixture()
    train, test = split_groups(groups, test_fraction=0.4, seed=42)
    assert len(train) == 3 and len(test) == 2
    assert {g.query_id for g in train}.isdisjoint(g.query_id for g in test)
    assert split_groups(groups, test_fraction=0.4, seed=42) == (train, test)
    assert {g.query_id for g in split_groups(tuple(reversed(groups)), seed=42)[1]} == {
        g.query_id for g in split_groups(groups, seed=42)[1]}


def test_split_invalid():
    groups = load_fixture()
    for fraction in (0, 1, -0.1):
        with pytest.raises(ValueError):
            split_groups(groups, test_fraction=fraction)
    with pytest.raises(ValueError):
        split_groups(groups[:1])
    with pytest.raises(ValueError):
        split_groups(groups + groups[:1])


def test_reject_duplicate_candidates(tmp_path):
    candidates = tmp_path / 'candidates.tsv'
    candidates.write_text('q1\tp1\nq1\tp1\n', encoding='utf-8')
    with pytest.raises(ValueError, match='duplicate'):
        load_groups(BASE / 'queries.tsv', BASE / 'passages.tsv',
                    candidates, BASE / 'qrels.tsv')


def test_reject_unknown_passage(tmp_path):
    candidates = tmp_path / 'candidates.tsv'
    candidates.write_text('q1\tmissing\n', encoding='utf-8')
    with pytest.raises(ValueError, match='unknown'):
        load_groups(BASE / 'queries.tsv', BASE / 'passages.tsv',
                    candidates, BASE / 'qrels.tsv')


def test_reject_bad_qrels(tmp_path):
    qrels = tmp_path / 'qrels.tsv'
    qrels.write_text('q1\tp1\t-1\n', encoding='utf-8')
    with pytest.raises(ValueError, match='negative'):
        load_groups(BASE / 'queries.tsv', BASE / 'passages.tsv',
                    BASE / 'candidates.tsv', qrels)
    qrels.write_text('q1\tp1\tfoo\n', encoding='utf-8')
    with pytest.raises(ValueError, match='integer'):
        load_groups(BASE / 'queries.tsv', BASE / 'passages.tsv',
                    BASE / 'candidates.tsv', qrels)


def test_reject_malformed_tsv(tmp_path):
    queries = tmp_path / 'queries.tsv'
    queries.write_text('q1\n', encoding='utf-8')
    with pytest.raises(ValueError, match='columns'):
        load_groups(queries, BASE / 'passages.tsv',
                    BASE / 'candidates.tsv', BASE / 'qrels.tsv')
