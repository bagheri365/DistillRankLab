"""Paired teacher-quality x student-feature-capacity synthetic ablations.

All capacities see the same queries, labels, and frozen teacher logits. A
capacity-k student sees the first k of three features; it is not a neural
network width/depth ablation. Teacher errors are sampled on the same queries
across error rates, but their corruption masks are not nested.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ablation import summarize as _summarize
from .comparison import Dataset, evaluate, train_student


@dataclass(frozen=True)
class CapacityConfig:
    capacities: tuple[int, ...] = (1, 2, 3)
    error_rates: tuple[float, ...] = (0.0, 0.1, 0.2, 0.4)
    temperatures: tuple[float, ...] = (1.0, 2.0, 4.0)
    seeds: tuple[int, ...] = (11, 12, 13, 14, 15)
    train_queries: int = 300
    test_queries: int = 1000
    steps: int = 500
    learning_rate: float = 0.15
    alpha: float = 0.5


FIELDS = ('capacity', 'error_rate', 'temperature', 'seed', 'method',
          'accuracy', 'teacher_agreement', 'delta_vs_hard')


def _validate(config: CapacityConfig) -> None:
    if not config.capacities or any(type(c) is not int or c not in (1, 2, 3)
                                    for c in config.capacities):
        raise ValueError('capacities must be integers from 1 to 3')
    if len(set(config.capacities)) != len(config.capacities):
        raise ValueError('capacities must be unique')
    if not config.error_rates or any(not np.isfinite(e) or not 0 <= e <= 1
                                     for e in config.error_rates):
        raise ValueError('error rates must be finite and in [0, 1]')
    if not config.temperatures or any(not np.isfinite(t) or t <= 0
                                      for t in config.temperatures):
        raise ValueError('temperatures must be positive and finite')
    if not config.seeds or any(type(s) is not int or s < 0 for s in config.seeds):
        raise ValueError('seeds must be nonnegative integers')
    if len(set(config.seeds)) != len(config.seeds):
        raise ValueError('seeds must be unique')
    if any(type(v) is not int or v <= 0 for v in
           (config.train_queries, config.test_queries, config.steps)):
        raise ValueError('query counts and steps must be positive integers')
    if not np.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError('learning rate must be positive and finite')
    if not np.isfinite(config.alpha) or not 0 <= config.alpha <= 1:
        raise ValueError('alpha must be finite and in [0, 1]')


def make_full_dataset(n_queries: int, *, seed: int,
                      teacher_error_rate: float = 0.0,
                      n_candidates: int = 4) -> Dataset:
    """Generate the same synthetic task as comparison.make_dataset, with 3 features.

    Retaining the third feature allows a fair capacity sweep without changing
    ground-truth labels or teacher scores between capacity settings.
    """
    if (type(n_queries) is not int or n_queries <= 0 or
            type(n_candidates) is not int or n_candidates < 2 or
            not np.isfinite(teacher_error_rate) or
            not 0 <= teacher_error_rate <= 1):
        raise ValueError('invalid dataset parameters')
    rng = np.random.default_rng(seed)
    full = rng.normal(size=(n_queries, n_candidates, 3))
    relevance = 1.5 * full[..., 0] - 0.6 * full[..., 1] + 0.8 * full[..., 2]
    labels = relevance.argmax(axis=1)
    teacher = relevance + rng.normal(scale=0.25, size=relevance.shape)
    corrupt = rng.random(n_queries) < teacher_error_rate
    for q in np.flatnonzero(corrupt):
        wrong = (int(labels[q]) + 1) % n_candidates
        teacher[q, wrong] = np.max(teacher[q]) + 3.0
    return Dataset(full, labels, teacher)


def _with_capacity(data: Dataset, capacity: int) -> Dataset:
    return Dataset(data.features[..., :capacity], data.labels,
                   data.teacher_logits)


def run_capacity_ablation(config: CapacityConfig = CapacityConfig()) -> list[dict]:
    """Evaluate all settings with paired seeds, datasets, and hard baselines.

    There are 4*3*5*3*3 = 540 output rows at the default settings, but
    hard-label models are trained only once per (error rate, seed, capacity).
    """
    _validate(config)
    rows = []
    for error_rate in config.error_rates:
        for seed in config.seeds:
            train_full = make_full_dataset(config.train_queries, seed=seed * 2,
                                           teacher_error_rate=error_rate)
            test_full = make_full_dataset(config.test_queries, seed=seed * 2 + 1,
                                          teacher_error_rate=error_rate)
            for capacity in config.capacities:
                train = _with_capacity(train_full, capacity)
                test = _with_capacity(test_full, capacity)
                hard_weights = train_student(train, mode='hard', steps=config.steps,
                                             learning_rate=config.learning_rate)
                baseline = evaluate(test, hard_weights)
                for temperature in config.temperatures:
                    for method in ('hard', 'soft', 'hybrid'):
                        if method == 'hard':
                            metrics = baseline
                        else:
                            weights = train_student(
                                train, mode=method, steps=config.steps,
                                learning_rate=config.learning_rate,
                                temperature=temperature, alpha=config.alpha)
                            metrics = evaluate(test, weights)
                        rows.append({
                            'capacity': capacity,
                            'error_rate': float(error_rate),
                            'temperature': float(temperature),
                            'seed': seed,
                            'method': method,
                            'accuracy': metrics['accuracy'],
                            'teacher_agreement': metrics['teacher_agreement'],
                            'delta_vs_hard': metrics['accuracy'] - baseline['accuracy'],
                        })
    return rows


def summarize_capacity(rows: list[dict]) -> list[dict]:
    """Aggregate mean and sample SD across seeds for each capacity and setting."""
    if not rows:
        raise ValueError('cannot summarize empty results')
    result = []
    for capacity in sorted({row['capacity'] for row in rows}):
        subset = [row for row in rows if row['capacity'] == capacity]
        for summary in _summarize(subset):
            result.append({'capacity': capacity, **summary})
    return result


def write_capacity_csv(path: str | Path, rows: list[dict]) -> None:
    """Export seed-level results. The destination directory must exist."""
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
