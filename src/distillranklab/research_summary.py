"""Validate and summarize the fixed 7E full experiment without retraining."""
from __future__ import annotations

from collections.abc import Mapping
import math

ARCHITECTURES = ('linear', 'mlp8', 'mlp16')
EXPECTED_SEEDS = (11, 23, 42, 67, 101)
PARAMETERS = {'linear': 8, 'mlp8': 81, 'mlp16': 161}


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def consolidate(report: Mapping) -> dict:
    """Reject incomplete/smoke reports and return an auditable 7E summary."""
    if not isinstance(report, Mapping) or tuple(report.get('seeds', ())) != EXPECTED_SEEDS:
        raise ValueError('full 7E report requires exactly the predeclared five seeds')
    if report.get('steps') != 120 or report.get('fit_queries') != 647 or report.get('validation_queries') != 162:
        raise ValueError('expected full 7E steps and query counts')
    if report.get('test_used') is not False or report.get('selected_architecture', 'missing') is not None:
        raise ValueError('7E must not use test or select architecture')
    if report.get('split_seed') != 42:
        raise ValueError('expected train-only split seed 42')
    if len(report.get('validation_query_ids', [])) != 162 or len(set(report['validation_query_ids'])) != 162:
        raise ValueError('invalid validation query IDs')
    trials = report.get('trials')
    if not isinstance(trials, list) or len(trials) != 15:
        raise ValueError('expected 15 individual training trials')
    observed = {(r.get('seed'), r.get('architecture')) for r in trials if isinstance(r, Mapping)}
    if observed != {(seed, arch) for seed in EXPECTED_SEEDS for arch in ARCHITECTURES}:
        raise ValueError('missing or duplicated seed-architecture trials')
    summaries = report.get('seed_summaries', {})
    output = {}
    for arch in ARCHITECTURES:
        entry = summaries.get(arch, {})
        if entry.get('parameters') != PARAMETERS[arch]:
            raise ValueError(f'invalid parameter count for {arch}')
        result = {'parameters': PARAMETERS[arch]}
        for key in ('mrr_at_10', 'ndcg_at_10', 'wall_seconds_fit_and_validation'):
            values = entry.get(key, {})
            if not all(_finite(values.get(field)) for field in ('mean', 'sample_std', 'min', 'max')):
                raise ValueError(f'invalid {arch} {key}')
            if values['sample_std'] < 0 or values['min'] > values['mean'] or values['mean'] > values['max']:
                raise ValueError(f'inconsistent {arch} {key}')
            observed_values = [r['validation_mrr_at_10' if key == 'mrr_at_10' else
                                 'validation_ndcg_at_10' if key == 'ndcg_at_10' else
                                 'wall_seconds_fit_and_validation'] for r in trials if r['architecture'] == arch]
            if not all(_finite(v) for v in observed_values):
                raise ValueError('invalid individual trial metric')
            if abs(sum(observed_values) / 5 - values['mean']) > 1e-8:
                raise ValueError('trial/summary mean mismatch')
            result[key] = {'mean': values['mean'], 'sample_std': values['sample_std']}
        output[arch] = result
    comparisons = {}
    for arch in ('mlp8', 'mlp16'):
        pair = report.get('paired_query_comparisons', {}).get(arch, {})
        if pair.get('comparison') != 'MLP minus linear' or pair.get('query_count') != 162 or pair.get('replicates') != 10000:
            raise ValueError(f'invalid paired comparison for {arch}')
        metrics = {}
        for name in ('mrr', 'ndcg'):
            metric = pair.get('metrics', {}).get(name, {})
            delta = metric.get('mean_candidate_minus_reference', metric.get('mean_teacher_minus_bm25'))
            ci = metric.get('bootstrap_percentile_ci')
            p = metric.get('paired_sign_flip_p_two_sided')
            if not (_finite(delta) and isinstance(ci, list) and len(ci) == 2 and
                    all(_finite(v) for v in ci) and ci[0] <= ci[1] and _finite(p) and 0 <= p <= 1):
                raise ValueError(f'invalid paired {arch} {name}')
            if abs(delta - (output[arch][name + '_at_10']['mean'] - output['linear'][name + '_at_10']['mean'])) > 1e-8:
                raise ValueError('paired delta does not match mean difference')
            metrics[name] = {'delta_candidate_minus_linear': delta, 'bootstrap_95_percent_ci': ci,
                             'sign_flip_p_two_sided': p}
        comparisons[arch] = metrics
    bm25 = report.get('bm25_validation_ndcg_at_10')
    if not _finite(bm25):
        raise ValueError('missing BM25 baseline')
    return {'experiment': '7E multi-seed robustness', 'split': 'official SciFact train-only validation',
            'fit_queries': 647, 'validation_queries': 162, 'seeds': list(EXPECTED_SEEDS),
            'steps': 120, 'bm25_validation_ndcg_at_10': bm25, 'architectures': output,
            'paired_mlp_minus_linear': comparisons, 'selected_architecture': None,
            'test_used': False,
            'interpretation': 'Exploratory reused validation; bootstrap conditional on five training seeds; no multiple-comparison correction; non-significant differences are not equivalence.'}


def markdown_summary(summary: Mapping) -> str:
    lines = ['# DistillRankLab — 7E research summary', '',
             'Official SciFact train-only validation: 647 fit / 162 validation queries; five seeds; 120 steps.',
             'Fixed BM25 top-100 candidates; unjudged passages treated as nonrelevant.', '',
             '| Architecture | Parameters | Mean MRR@10 | Mean nDCG@10 ± seed SD | Fit + validation (s) |',
             '|---|---:|---:|---:|---:|']
    for arch, data in summary['architectures'].items():
        lines.append(f"| {arch} | {data['parameters']} | {data['mrr_at_10']['mean']:.6f} | "
                     f"{data['ndcg_at_10']['mean']:.6f} ± {data['ndcg_at_10']['sample_std']:.6f} | "
                     f"{data['wall_seconds_fit_and_validation']['mean']:.2f} |")
    lines.extend(['', f"BM25 validation nDCG@10: **{summary['bm25_validation_ndcg_at_10']:.6f}**.", '',
                  '## Paired MLP minus linear (query-level, seed-averaged)', '',
                  '| Model | Metric | Mean delta | 95% bootstrap CI | Sign-flip p |',
                  '|---|---|---:|---|---:|'])
    for arch, metrics in summary['paired_mlp_minus_linear'].items():
        for name, data in metrics.items():
            low, high = data['bootstrap_95_percent_ci']
            lines.append(f"| {arch} | {name.upper()}@10 | {data['delta_candidate_minus_linear']:+.6f} | "
                         f"[{low:.6f}, {high:.6f}] | {data['sign_flip_p_two_sided']:.4f} |")
    lines.extend(['', '**Limitations:** Reused validation is exploratory; query bootstrap is conditional on evaluated seeds; '
                  'p-values are uncorrected for multiple comparisons; no independent confirmation or equivalence claim.',
                  'No test split was accessed by this consolidation.', ''])
    return '\n'.join(lines)
