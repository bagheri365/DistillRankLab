"""Regression tests for paired teacher-quality x student-capacity experiments."""

import csv

import numpy as np
import pytest

from distillranklab.comparison import make_dataset
from distillranklab.capacity import (CapacityConfig, make_full_dataset,
                                     run_capacity_ablation, summarize_capacity,
                                     write_capacity_csv)


def small(**overrides):
    values = dict(capacities=(1, 2, 3), error_rates=(0., .2),
                  temperatures=(1., 2.), seeds=(3, 4),
                  train_queries=24, test_queries=30, steps=15)
    values.update(overrides)
    return CapacityConfig(**values)


def test_full_dataset_matches_existing_generator():
    for error in (0., .2, .4):
        full = make_full_dataset(20, seed=7, teacher_error_rate=error)
        old = make_dataset(20, seed=7, teacher_error_rate=error)
        np.testing.assert_array_equal(full.features[..., :2], old.features)
        np.testing.assert_array_equal(full.labels, old.labels)
        np.testing.assert_array_equal(full.teacher_logits, old.teacher_logits)


def test_capacity_slices_preserve_underlying_data():
    data = make_full_dataset(10, seed=9)
    assert data.features.shape == (10, 4, 3)
    assert np.isfinite(data.features).all()


def test_run_count_and_methods():
    rows = run_capacity_ablation(small())
    assert len(rows) == 3 * 2 * 2 * 2 * 3
    assert {r['method'] for r in rows} == {'hard', 'soft', 'hybrid'}


def test_deterministic():
    assert run_capacity_ablation(small()) == run_capacity_ablation(small())


def test_paired_deltas_and_temperature_invariant_hard():
    rows = run_capacity_ablation(small())
    for row in rows:
        baseline = next(r for r in rows if r['method'] == 'hard'
                        and all(r[k] == row[k] for k in
                                ('capacity', 'error_rate', 'seed', 'temperature')))
        assert np.isclose(row['delta_vs_hard'],
                          row['accuracy'] - baseline['accuracy'])
        assert 0 <= row['accuracy'] <= 1
        assert 0 <= row['teacher_agreement'] <= 1
    for capacity in (1, 2, 3):
        for error in (0., .2):
            for seed in (3, 4):
                hard = [r for r in rows if r['method'] == 'hard'
                        and r['capacity'] == capacity and r['error_rate'] == error
                        and r['seed'] == seed]
                assert len(hard) == 2
                assert hard[0]['accuracy'] == hard[1]['accuracy']
                assert all(r['delta_vs_hard'] == 0 for r in hard)


def test_summary():
    summary = summarize_capacity(run_capacity_ablation(small()))
    assert len(summary) == 3 * 2 * 2 * 3
    assert all(r['n_seeds'] == 2 for r in summary)
    assert all(r['delta_vs_hard_mean'] == 0 for r in summary
               if r['method'] == 'hard')


def test_single_seed_summary_sd():
    summary = summarize_capacity(run_capacity_ablation(small(seeds=(3,))))
    assert all(r['accuracy_sd'] == 0 for r in summary)


def test_empty_summary():
    with pytest.raises(ValueError):
        summarize_capacity([])


def test_csv_roundtrip(tmp_path):
    rows = run_capacity_ablation(small(seeds=(3,)))
    path = tmp_path / 'capacity.csv'
    write_capacity_csv(path, rows)
    with path.open(newline='', encoding='utf-8') as handle:
        loaded = list(csv.DictReader(handle))
    assert len(loaded) == len(rows)
    assert int(loaded[0]['capacity']) == rows[0]['capacity']


@pytest.mark.parametrize('kwargs', [
    {'capacities': ()}, {'capacities': (0,)}, {'capacities': (4,)},
    {'capacities': (1, 1)}, {'capacities': (1.0,)},
    {'error_rates': ()}, {'error_rates': (-.1,)}, {'error_rates': (float('nan'),)},
    {'temperatures': ()}, {'temperatures': (0,)},
    {'seeds': ()}, {'seeds': (-1,)}, {'seeds': (3, 3)},
    {'train_queries': 0}, {'test_queries': 0}, {'steps': 0},
    {'learning_rate': 0}, {'alpha': 1.1},
])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        run_capacity_ablation(small(**kwargs))


@pytest.mark.parametrize('kwargs', [
    {'n_queries': 0}, {'n_queries': 10, 'teacher_error_rate': -0.1},
    {'n_queries': 10, 'n_candidates': 1},
])
def test_invalid_dataset(kwargs):
    with pytest.raises(ValueError):
        make_full_dataset(seed=1, **kwargs)
