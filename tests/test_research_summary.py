import copy

import pytest

from distillranklab.research_summary import consolidate, markdown_summary


def sample_report():
    seeds = [11, 23, 42, 67, 101]
    values = {'linear': (0.65, 0.68, 8), 'mlp8': (0.652, 0.682, 81),
              'mlp16': (0.651, 0.681, 161)}
    trials = []
    summaries = {}
    for arch, (mrr, ndcg, params) in values.items():
        for seed in seeds:
            trials.append({'seed': seed, 'architecture': arch,
                           'validation_mrr_at_10': mrr, 'validation_ndcg_at_10': ndcg,
                           'wall_seconds_fit_and_validation': 10.0})
        summaries[arch] = {'parameters': params, 'mrr_at_10': {'mean': mrr, 'sample_std': 0., 'min': mrr, 'max': mrr},
                           'ndcg_at_10': {'mean': ndcg, 'sample_std': 0., 'min': ndcg, 'max': ndcg},
                           'wall_seconds_fit_and_validation': {'mean': 10., 'sample_std': 0., 'min': 10., 'max': 10.}}
    pairs = {}
    for arch in ('mlp8', 'mlp16'):
        pairs[arch] = {'comparison': 'MLP minus linear', 'query_count': 162, 'replicates': 10000,
                       'metrics': {metric: {'mean_teacher_minus_bm25': values[arch][idx] - values['linear'][idx],
                                            'bootstrap_percentile_ci': [-0.01, 0.02],
                                            'paired_sign_flip_p_two_sided': 0.5}
                                   for idx, metric in enumerate(('mrr', 'ndcg'))}}
    return {'seeds': seeds, 'steps': 120, 'fit_queries': 647, 'validation_queries': 162,
            'test_used': False, 'selected_architecture': None, 'split_seed': 42,
            'validation_query_ids': [str(i) for i in range(162)],
            'trials': trials, 'seed_summaries': summaries, 'paired_query_comparisons': pairs,
            'bm25_validation_ndcg_at_10': 0.688365}


def test_consolidation_and_markdown():
    summary = consolidate(sample_report())
    assert summary['architectures']['mlp8']['parameters'] == 81
    assert summary['paired_mlp_minus_linear']['mlp8']['ndcg']['delta_candidate_minus_linear'] == pytest.approx(0.002)
    rendered = markdown_summary(summary)
    assert '0.688365' in rendered and 'exploratory' in rendered.lower()
    assert 'mlp16' in rendered
    assert consolidate(sample_report()) == summary


@pytest.mark.parametrize('change', [
    lambda r: r.update(seeds=[11, 42]),
    lambda r: r.update(steps=3),
    lambda r: r.update(test_used=True),
    lambda r: r.update(split_seed=99),
    lambda r: r['validation_query_ids'].append('extra'),
    lambda r: r['trials'].pop(),
    lambda r: r['trials'][0].update(seed=42),
    lambda r: r['seed_summaries']['mlp8']['ndcg_at_10'].update(mean=0.9),
    lambda r: r['paired_query_comparisons']['mlp8']['metrics']['ndcg'].update(mean_teacher_minus_bm25=0.9),
])
def test_reject_invalid_reports(change):
    report = sample_report()
    change(report)
    with pytest.raises(ValueError):
        consolidate(report)


def test_neutral_alias_preferred():
    report = sample_report()
    report['paired_query_comparisons']['mlp8']['metrics']['ndcg']['mean_candidate_minus_reference'] = 0.002
    assert consolidate(report)['paired_mlp_minus_linear']['mlp8']['ndcg']['delta_candidate_minus_linear'] == pytest.approx(0.002)
