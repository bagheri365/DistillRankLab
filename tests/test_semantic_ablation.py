"""8C unit tests are offline and use no pretrained model downloads."""
import numpy as np
import pytest

from distillranklab.real_data import Candidate, QueryGroup
from distillranklab.semantic_ablation import (FEATURE_BLOCKS, compare_paired,
                                                masked_features, run_feature_ablation)


def groups():
    return tuple(QueryGroup(f'q{i}', f'query {i}', (
        Candidate('a', 'positive passage', 1),
        Candidate('b', 'other passage', None))) for i in range(3))


def features():
    return [np.array([[1., 0., 0., 0., 0., 0., 0., 0., 1., .5],
                      [0., 1., 0., 0., 0., 0., 0., 0., .1, .2]]) for _ in groups()]


def test_feature_blocks_have_correct_dimensions():
    assert {k: len(v) for k, v in FEATURE_BLOCKS.items()} == {
        'lexical_only': 8, 'semantic_only': 2, 'combined': 10}


@pytest.mark.parametrize('block,zeros', [('lexical_only', slice(8, 10)),
                                         ('semantic_only', slice(0, 8))])
def test_masking_does_not_mutate_input(block, zeros):
    original = features()
    out = masked_features(original, block)
    assert np.all(out[0][:, zeros] == 0)
    np.testing.assert_allclose(original[0][0, -2:], [1, .5])
    assert not np.shares_memory(original[0], out[0])


def test_combined_preserves_all_coordinates():
    np.testing.assert_array_equal(masked_features(features(), 'combined')[0], features()[0])


@pytest.mark.parametrize('bad', ['other', '', 'lexical'])
def test_unknown_block_rejected(bad):
    with pytest.raises(ValueError):
        masked_features(features(), bad)


@pytest.mark.parametrize('bad', [np.zeros((2, 9)), np.array([[float('nan')] * 10])])
def test_invalid_feature_matrix_rejected(bad):
    with pytest.raises(ValueError):
        masked_features([bad], 'combined')


def test_ablation_train_predict_and_counts():
    gs = groups()
    teacher = {(g.query_id, c.passage_id): float(c.passage_id == 'a')
               for g in gs for c in g.candidates}
    report, predictions = run_feature_ablation(gs, gs, features(), features(), teacher, steps=3)
    assert set(report) == set(FEATURE_BLOCKS)
    assert [report[b]['effective_parameters'] for b in FEATURE_BLOCKS] == [8, 2, 10]
    assert all(len(predictions[b]) == 6 for b in FEATURE_BLOCKS)
    assert all(report[b]['metrics']['total_queries'] == 3 for b in FEATURE_BLOCKS)


def test_ablation_rejects_mismatched_groups():
    with pytest.raises(ValueError):
        run_feature_ablation(groups(), groups(), features()[:1], features(), {})


def test_paired_identity_zero_delta():
    gs = groups()
    scores = {(g.query_id, c.passage_id): float(c.passage_id == 'a')
              for g in gs for c in g.candidates}
    result = compare_paired(gs, scores, scores, baseline_name='a', challenger_name='b', replicates=100)
    assert result['difference'] == 'b minus a'
    assert result['metrics']['ndcg']['mean_challenger_minus_baseline'] == 0
    assert result['metrics']['ndcg']['bootstrap_percentile_ci'] == [0, 0]
    assert result['metrics']['ndcg']['paired_sign_flip_p_two_sided'] == 1


def test_paired_direction_is_challenger_minus_baseline():
    gs = groups()
    good = {(g.query_id, c.passage_id): float(c.passage_id == 'a')
            for g in gs for c in g.candidates}
    bad = {(g.query_id, c.passage_id): float(c.passage_id == 'b')
           for g in gs for c in g.candidates}
    result = compare_paired(gs, bad, good, baseline_name='bad', challenger_name='good', replicates=100)
    assert result['metrics']['mrr']['mean_challenger_minus_baseline'] == .5
    assert result['metrics']['ndcg']['mean_challenger_minus_baseline'] > 0
