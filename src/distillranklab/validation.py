"""Query-disjoint, validation-only selection for the SciFact linear student.

All candidate pools are fixed BM25 pools. No test data enters selection.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product
import math

from .benchmark import bm25_order_scores, summarize_ranking
from .real_data import QueryGroup, split_groups
from .real_student import evaluate_student, fit_student


def make_configs(*, learning_rates: Sequence[float] = (0.05,),
                 temperatures: Sequence[float] = (1.0, 2.0),
                 hard_weights: Sequence[float] = (0.25, 0.5),
                 steps: Sequence[int] = (120,)) -> list[dict]:
    """Construct an explicit, deterministic hard/soft/hybrid search grid."""
    if (not learning_rates or not temperatures or not hard_weights or not steps
            or any(not math.isfinite(v) or v <= 0 for v in learning_rates)
            or any(not math.isfinite(v) or v <= 0 for v in temperatures)
            or any(not math.isfinite(v) or not 0 < v < 1 for v in hard_weights)
            or any(type(v) is not int or v < 1 for v in steps)):
        raise ValueError('invalid hyperparameter grid')
    configs = []
    for lr, count in product(learning_rates, steps):
        configs.append(dict(mode='hard', learning_rate=lr, steps=count,
                            temperature=2.0, hard_weight=1.0))
        for temp in temperatures:
            configs.append(dict(mode='soft', learning_rate=lr, steps=count,
                                temperature=temp, hard_weight=0.0))
            for weight in hard_weights:
                configs.append(dict(mode='hybrid', learning_rate=lr, steps=count,
                                    temperature=temp, hard_weight=weight))
    return configs


def select_student(groups: Sequence[QueryGroup],
                   teacher_scores: Mapping[tuple[str, str], float] | None, *,
                   configs: Sequence[dict] | None = None,
                   validation_fraction: float = 0.2,
                   split_seed: int = 42, training_seed: int = 42,
                   k: int = 10) -> tuple[dict, dict]:
    """Select by validation all-query nDCG@k; return report and refit model.

    Refit uses all official train queries only, after selecting hyperparameters.
    No held-out test data is accepted by this function.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError('k must be positive')
    configs = list(make_configs() if configs is None else configs)
    if not configs:
        raise ValueError('grid must be nonempty')
    train, validation = split_groups(tuple(groups), test_fraction=validation_fraction,
                                     seed=split_seed)
    rows = []
    best_index = None
    best_value = float('-inf')
    for index, config in enumerate(configs):
        # Only documented fit_student parameters are accepted; fail on typos.
        if set(config) != {'mode', 'learning_rate', 'steps', 'temperature', 'hard_weight'}:
            raise ValueError('invalid config keys')
        model = fit_student(train, teacher_scores, seed=training_seed, **config)
        metrics = evaluate_student(validation, model, k=k)
        value = metrics['ndcg_at_k_all_queries']
        rows.append({'index': index, 'config': dict(config),
                     'validation_mrr_at_k_all_queries': metrics['mrr_at_k_all_queries'],
                     'validation_ndcg_at_k_all_queries': value})
        if value > best_value + 1e-12:
            best_value, best_index = value, index
    chosen = configs[best_index]
    final_model = fit_student(groups, teacher_scores, seed=training_seed, **chosen)
    report = {
        'selection_metric': f'validation_all_query_ndcg@{k}',
        'validation_fraction': validation_fraction, 'split_seed': split_seed,
        'training_seed': training_seed, 'fit_queries': len(train),
        'validation_queries': len(validation), 'refit_queries': len(groups),
        'fit_query_ids': sorted(g.query_id for g in train),
        'validation_query_ids': sorted(g.query_id for g in validation),
        'validation_bm25': summarize_ranking(validation, bm25_order_scores(validation), k=k),
        'trials': rows, 'selected_index': best_index, 'selected_config': dict(chosen),
        'selected_validation_ndcg_at_k_all_queries': best_value,
        'tie_break': 'first configuration in supplied order within 1e-12',
        'refit_note': 'Selected configuration refit on all official train queries; validation scores are from pre-refit model',
    }
    return report, final_model
