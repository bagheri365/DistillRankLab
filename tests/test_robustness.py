import numpy as np
import pytest

from distillranklab.real_data import Candidate, QueryGroup
from distillranklab.robustness import query_paired_summary, run_robustness, summarize_seeds


def groups(prefix):
    return tuple(QueryGroup(prefix + str(i), 'alpha beta', (
        Candidate(prefix + str(i) + 'a', 'alpha beta', 1),
        Candidate(prefix + str(i) + 'b', 'other unrelated', None),
        Candidate(prefix + str(i) + 'c', 'alpha', None))) for i in range(3))


def teacher(groups):
    return {(g.query_id, c.passage_id): float(3-j) for g in groups for j,c in enumerate(g.candidates)}


def test_summarize_seed_sample_std():
    result = summarize_seeds([1, 2, 3])
    assert result['mean'] == 2 and result['sample_std'] == pytest.approx(1)
    assert summarize_seeds([4])['sample_std'] is None


@pytest.mark.parametrize('values', [[], [float('nan')], [float('inf')]])
def test_bad_seed_values(values):
    with pytest.raises(ValueError): summarize_seeds(values)


def test_query_paired_deterministic():
    rows = [{'query_id': str(i), 'linear_mrr': 0.2, 'mlp_mrr': 0.4,
             'linear_ndcg': 0.3, 'mlp_ndcg': 0.6} for i in range(5)]
    a = query_paired_summary(rows, replicates=100, seed=3)
    assert a == query_paired_summary(rows, replicates=100, seed=3)
    assert a['metrics']['ndcg']['mean_teacher_minus_bm25'] == pytest.approx(0.3)
    assert a['metrics']['mrr']['bootstrap_percentile_ci'] == pytest.approx([0.2, 0.2])


def test_query_paired_rejects_duplicate_ids():
    row = {'query_id': 'a', 'linear_mrr': 0.1, 'mlp_mrr': 0.2,
           'linear_ndcg': 0.1, 'mlp_ndcg': 0.2}
    with pytest.raises(ValueError): query_paired_summary([row, row], replicates=100)


def test_run_is_reproducible_except_timing():
    fit, val = groups('train'), groups('valid')
    a = run_robustness(fit, val, teacher(fit), seeds=[3, 5], steps=3, replicates=100)
    b = run_robustness(fit, val, teacher(fit), seeds=[3, 5], steps=3, replicates=100)
    assert a['seed_summaries']['mlp8']['ndcg_at_10'] == b['seed_summaries']['mlp8']['ndcg_at_10']
    assert a['paired_query_comparisons'] == b['paired_query_comparisons']
    assert len(a['trials']) == 6
    assert a['selected_architecture'] is None
    assert a['test_used'] is False
    assert a['seed_summaries']['mlp16']['parameters'] == 161


@pytest.mark.parametrize('seeds', [[], [1, 1], [-1], [True], [1.5]])
def test_reject_bad_seeds(seeds):
    fit, val = groups('train'), groups('valid')
    with pytest.raises(ValueError): run_robustness(fit, val, teacher(fit), seeds=seeds, steps=2, replicates=100)


def test_reject_overlap():
    fit = groups('train')
    with pytest.raises(ValueError): run_robustness(fit, fit, teacher(fit), seeds=[1], steps=2, replicates=100)


def test_reject_steps():
    fit, val = groups('train'), groups('valid')
    with pytest.raises(ValueError): run_robustness(fit, val, teacher(fit), seeds=[1], steps=0, replicates=100)
