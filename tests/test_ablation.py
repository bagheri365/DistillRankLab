"""Tests for the reproducible multi-seed ablation study."""

import csv

import numpy as np
import pytest

from distillranklab.ablation import AblationConfig, run_ablation, summarize, write_csv


def tiny_config(**kwargs):
    defaults = dict(error_rates=(0.0, 0.2), temperatures=(1.0, 2.0),
                    seeds=(3, 4), train_queries=24, test_queries=30, steps=15)
    defaults.update(kwargs)
    return AblationConfig(**defaults)


def test_expected_number_of_runs():
    rows = run_ablation(tiny_config())
    assert len(rows) == 2 * 2 * 2 * 3
    assert set(r['method'] for r in rows) == {'hard', 'soft', 'hybrid'}


def test_deterministic():
    config = tiny_config()
    assert run_ablation(config) == run_ablation(config)


def test_hard_baseline_is_temperature_invariant():
    rows = run_ablation(tiny_config())
    for error in (0.0, 0.2):
        for seed in (3, 4):
            hard = [r for r in rows if r['method'] == 'hard'
                    and r['error_rate'] == error and r['seed'] == seed]
            assert len(hard) == 2
            assert hard[0]['accuracy'] == hard[1]['accuracy']
            assert all(r['delta_vs_hard'] == 0 for r in hard)


def test_paired_deltas():
    rows = run_ablation(tiny_config())
    for row in rows:
        baseline = next(r for r in rows if r['method'] == 'hard'
                        and r['seed'] == row['seed']
                        and r['error_rate'] == row['error_rate']
                        and r['temperature'] == row['temperature'])
        assert np.isclose(row['delta_vs_hard'], row['accuracy'] - baseline['accuracy'])
        assert 0 <= row['accuracy'] <= 1
        assert 0 <= row['teacher_agreement'] <= 1


def test_summary_sample_sd_and_group_count():
    summary = summarize(run_ablation(tiny_config()))
    assert len(summary) == 2 * 2 * 3
    assert all(r['n_seeds'] == 2 for r in summary)
    assert all(r['accuracy_sd'] >= 0 for r in summary)
    assert all(r['delta_vs_hard_mean'] == 0 for r in summary if r['method'] == 'hard')


def test_single_seed_sd_is_zero():
    summary = summarize(run_ablation(tiny_config(seeds=(3,))))
    assert all(r['accuracy_sd'] == 0 for r in summary)


def test_csv_roundtrip(tmp_path):
    rows = run_ablation(tiny_config(seeds=(3,)))
    path = tmp_path / 'results.csv'
    write_csv(path, rows)
    with path.open(newline='', encoding='utf-8') as handle:
        loaded = list(csv.DictReader(handle))
    assert len(loaded) == len(rows)
    assert float(loaded[0]['accuracy']) == rows[0]['accuracy']


@pytest.mark.parametrize('overrides', [
    {'error_rates': ()}, {'error_rates': (-0.1,)}, {'error_rates': (1.1,)},
    {'temperatures': (0,)}, {'temperatures': (float('nan'),)},
    {'seeds': ()}, {'seeds': (2, 2)}, {'seeds': (-1,)},
    {'train_queries': 0}, {'test_queries': 0}, {'steps': 0},
    {'learning_rate': 0}, {'alpha': 1.1},
])
def test_invalid_configs(overrides):
    with pytest.raises(ValueError):
        run_ablation(tiny_config(**overrides))


def test_empty_summary_rejected():
    with pytest.raises(ValueError):
        summarize([])
