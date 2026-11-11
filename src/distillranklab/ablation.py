"""Reproducible, paired ablations for synthetic ranking distillation."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .comparison import evaluate, make_dataset, train_student


@dataclass(frozen=True)
class AblationConfig:
    error_rates: tuple[float, ...] = (0.0, 0.1, 0.2, 0.4)
    temperatures: tuple[float, ...] = (1.0, 2.0, 4.0)
    seeds: tuple[int, ...] = (11, 12, 13, 14, 15)
    train_queries: int = 300
    test_queries: int = 1000
    steps: int = 500
    learning_rate: float = 0.15
    alpha: float = 0.5


FIELDS = ('error_rate', 'temperature', 'seed', 'method', 'accuracy',
          'teacher_agreement', 'delta_vs_hard')


def _validate(config: AblationConfig) -> None:
    if not config.error_rates or not config.temperatures or not config.seeds:
        raise ValueError('error rates, temperatures, and seeds must be nonempty')
    if any(not np.isfinite(v) or not 0 <= v <= 1 for v in config.error_rates):
        raise ValueError('error rates must be finite and in [0, 1]')
    if any(not np.isfinite(v) or v <= 0 for v in config.temperatures):
        raise ValueError('temperatures must be positive and finite')
    if any(not isinstance(s, (int, np.integer)) or s < 0 for s in config.seeds):
        raise ValueError('seeds must be nonnegative integers')
    if len(set(config.seeds)) != len(config.seeds):
        raise ValueError('seeds must be unique for independent replicates')
    if config.train_queries <= 0 or config.test_queries <= 0 or config.steps <= 0:
        raise ValueError('dataset sizes and steps must be positive')
    if not np.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError('learning rate must be positive and finite')
    if not np.isfinite(config.alpha) or not 0 <= config.alpha <= 1:
        raise ValueError('alpha must be in [0, 1]')


def run_ablation(config: AblationConfig = AblationConfig()) -> list[dict]:
    """Run paired comparisons; the hard baseline is shared across temperatures.

    Training and testing use distinct deterministic seeds. Within each
    (error_rate, seed) group, all methods use identical datasets. Teacher
    corruption is sampled separately for each error rate, so error-rate
    comparisons are descriptive rather than strictly nested interventions.
    """
    _validate(config)
    rows = []
    for error_rate in config.error_rates:
        for seed in config.seeds:
            train = make_dataset(config.train_queries, seed=int(seed) * 2,
                                 teacher_error_rate=error_rate)
            test = make_dataset(config.test_queries, seed=int(seed) * 2 + 1,
                                teacher_error_rate=error_rate)
            baseline = evaluate(test, train_student(
                train, mode='hard', steps=config.steps,
                learning_rate=config.learning_rate))
            for temperature in config.temperatures:
                for mode in ('hard', 'soft', 'hybrid'):
                    if mode == 'hard':
                        metrics = baseline
                    else:
                        weights = train_student(
                            train, mode=mode, steps=config.steps,
                            learning_rate=config.learning_rate,
                            temperature=temperature, alpha=config.alpha)
                        metrics = evaluate(test, weights)
                    rows.append({
                        'error_rate': float(error_rate),
                        'temperature': float(temperature),
                        'seed': int(seed),
                        'method': mode,
                        'accuracy': metrics['accuracy'],
                        'teacher_agreement': metrics['teacher_agreement'],
                        'delta_vs_hard': metrics['accuracy'] - baseline['accuracy'],
                    })
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """Mean and sample SD across seeds, grouped by error/temperature/method."""
    if not rows:
        raise ValueError('cannot summarize empty results')
    groups = {}
    for row in rows:
        key = (row['error_rate'], row['temperature'], row['method'])
        groups.setdefault(key, []).append(row)
    summary = []
    for (error_rate, temperature, method), group in sorted(groups.items()):
        result = {'error_rate': error_rate, 'temperature': temperature,
                  'method': method, 'n_seeds': len(group)}
        for field in ('accuracy', 'teacher_agreement', 'delta_vs_hard'):
            values = np.asarray([r[field] for r in group], dtype=float)
            result[f'{field}_mean'] = float(values.mean())
            result[f'{field}_sd'] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(result)
    return summary


def write_csv(path: str | Path, rows: list[dict]) -> None:
    """Write raw, seed-level results; parent directory must already exist."""
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
