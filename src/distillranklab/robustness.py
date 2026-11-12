"""Multi-seed student robustness, with separate seed and query uncertainty."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from time import perf_counter

import numpy as np

from .benchmark import bm25_order_scores, per_query_comparison, summarize_ranking
from .nonlinear_student import fit_nonlinear, predict_nonlinear, parameter_count
from .real_data import QueryGroup
from .real_student import fit_student, predict_student
from .statistics import paired_statistics

ARCHITECTURES = ('linear', 'mlp8', 'mlp16')


def summarize_seeds(values: Sequence[float]) -> dict:
    """Descriptive seed variability; sample SD, not a query-level confidence interval."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or not len(arr) or not np.isfinite(arr).all():
        raise ValueError('expected nonempty finite seed values')
    return {'mean': float(arr.mean()), 'sample_std': float(arr.std(ddof=1)) if len(arr) > 1 else None,
            'min': float(arr.min()), 'max': float(arr.max())}


def query_paired_summary(query_rows: Sequence[dict], *, replicates: int = 10000,
                         seed: int = 42) -> dict:
    """Paired query uncertainty for precomputed MLP-minus-linear query scores.

    When rows are averaged across training seeds, the bootstrap treats the
    trained seed ensemble as fixed; it does NOT quantify training-seed uncertainty.
    """
    if not query_rows:
        raise ValueError('query rows required')
    rows = []
    for row in query_rows:
        rows.append({'query_id': row['query_id'],
                     'bm25_mrr_at_k': row['linear_mrr'],
                     'teacher_mrr_at_k': row['mlp_mrr'],
                     'delta_mrr_at_k': row['mlp_mrr'] - row['linear_mrr'],
                     'bm25_ndcg_at_k': row['linear_ndcg'],
                     'teacher_ndcg_at_k': row['mlp_ndcg'],
                     'delta_ndcg_at_k': row['mlp_ndcg'] - row['linear_ndcg']})
    result = paired_statistics({'queries': rows, 'query_count': len(rows),
                                'k': 10, 'unjudged': 'nonrelevant'},
                               replicates=replicates, seed=seed)
    return {'sampling_unit': 'query', 'comparison': 'MLP minus linear',
            'mean_across_training_seeds_first': True,
            'interpretation': 'Conditional on the evaluated seeds; exploratory repeated validation; not seed uncertainty',
            'metrics': result['metrics'], 'replicates': replicates, 'seed': seed,
            'query_count': len(rows)}


def run_robustness(fit: Sequence[QueryGroup], validation: Sequence[QueryGroup],
                   teacher_scores: Mapping[tuple[str, str], float], *,
                   seeds: Sequence[int] = (11, 23, 42, 67, 101),
                   steps: int = 120, replicates: int = 10000,
                   bootstrap_seed: int = 42) -> dict:
    """Fixed three-model comparison on query-disjoint train-only validation.

    Model hyperparameters are fixed before the experiment. No model is selected
    and no official SciFact test data are loaded by this function.
    """
    if not fit or not validation or set(g.query_id for g in fit) & set(g.query_id for g in validation):
        raise ValueError('fit/validation must be nonempty and query-disjoint')
    if (not seeds or any(type(s) is not int or s < 0 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('seeds must be distinct nonnegative integers')
    if type(steps) is not int or steps < 1:
        raise ValueError('steps must be positive')
    sorted_validation = tuple(sorted(validation, key=lambda g: g.query_id))
    query_ids = [g.query_id for g in sorted_validation]
    records = []
    per_seed_rows = {}
    for training_seed in seeds:
        scores_by_arch = {}
        for architecture in ARCHITECTURES:
            start = perf_counter()
            common = dict(mode='hybrid', feature_set='rich', hard_weight=0.5,
                          temperature=2.0, learning_rate=0.05, steps=steps, seed=training_seed)
            if architecture == 'linear':
                model = fit_student(fit, teacher_scores, **common)
                scores = predict_student(sorted_validation, model)
                params = len(model['weights'])
            else:
                hidden = 8 if architecture == 'mlp8' else 16
                model = fit_nonlinear(fit, teacher_scores, hidden_units=hidden, **common)
                scores = predict_nonlinear(sorted_validation, model)
                params = parameter_count(8, hidden)
            metrics = summarize_ranking(sorted_validation, scores, k=10)
            records.append({'seed': training_seed, 'architecture': architecture,
                            'parameters': params, 'wall_seconds_fit_and_validation': perf_counter()-start,
                            'validation_mrr_at_10': metrics['mrr_at_k_all_queries'],
                            'validation_ndcg_at_10': metrics['ndcg_at_k_all_queries']})
            scores_by_arch[architecture] = scores
        baseline = scores_by_arch['linear']
        per_seed_rows[training_seed] = {}
        for arch in ('mlp8', 'mlp16'):
            comparison = per_query_comparison(sorted_validation, baseline, scores_by_arch[arch], k=10)
            per_seed_rows[training_seed][arch] = comparison['queries']
    summaries = {}
    for arch in ARCHITECTURES:
        selected = [r for r in records if r['architecture'] == arch]
        summaries[arch] = {'parameters': selected[0]['parameters'],
                           'mrr_at_10': summarize_seeds([r['validation_mrr_at_10'] for r in selected]),
                           'ndcg_at_10': summarize_seeds([r['validation_ndcg_at_10'] for r in selected]),
                           'wall_seconds_fit_and_validation': summarize_seeds([r['wall_seconds_fit_and_validation'] for r in selected])}
    paired = {}
    for arch in ('mlp8', 'mlp16'):
        averaged = []
        for i, qid in enumerate(query_ids):
            rows = [per_seed_rows[s][arch][i] for s in seeds]
            if any(r['query_id'] != qid for r in rows):
                raise RuntimeError('query alignment failure')
            averaged.append({'query_id': qid,
                             'linear_mrr': float(np.mean([r['bm25_mrr_at_k'] for r in rows])),
                             'mlp_mrr': float(np.mean([r['teacher_mrr_at_k'] for r in rows])),
                             'linear_ndcg': float(np.mean([r['bm25_ndcg_at_k'] for r in rows])),
                             'mlp_ndcg': float(np.mean([r['teacher_ndcg_at_k'] for r in rows]))})
        paired[arch] = query_paired_summary(averaged, replicates=replicates, seed=bootstrap_seed)
    bm25 = summarize_ranking(sorted_validation, bm25_order_scores(sorted_validation), k=10)
    return {'seeds': list(seeds), 'steps': steps, 'fit_queries': len(fit),
            'validation_queries': len(validation), 'validation_query_ids': query_ids,
            'bm25_validation_mrr_at_10': bm25['mrr_at_k_all_queries'],
            'bm25_validation_ndcg_at_10': bm25['ndcg_at_k_all_queries'],
            'trials': records, 'seed_summaries': summaries,
            'paired_query_comparisons': paired, 'selected_architecture': None,
            'test_used': False,
            'limitations': 'Exploratory reuse of validation; query bootstrap conditional on evaluated seeds; seed SD is descriptive; no multiple-comparison correction; unjudged assumed nonrelevant'}
