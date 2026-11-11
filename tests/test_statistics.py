
import pytest

from distillranklab.statistics import paired_statistics


def fixture():
    rows = []
    for i, (b, t) in enumerate(((0, 1), (0.5, 1), (1, 0.5), (0, 0))):
        rows.append({'query_id': str(i), 'bm25_mrr_at_k': b, 'teacher_mrr_at_k': t,
                     'delta_mrr_at_k': t-b, 'bm25_ndcg_at_k': b,
                     'teacher_ndcg_at_k': t, 'delta_ndcg_at_k': t-b})
    return {'query_count': 4, 'k': 10, 'unjudged': 'nonrelevant', 'queries': rows}


def test_deterministic_and_point_estimate():
    a = paired_statistics(fixture(), replicates=200, seed=7)
    assert a == paired_statistics(fixture(), replicates=200, seed=7)
    assert a['metrics']['mrr']['mean_teacher_minus_bm25'] == pytest.approx(0.25)
    assert a['metrics']['ndcg']['mean_teacher_minus_bm25'] == pytest.approx(0.25)
    assert a['query_count'] == 4


def test_zero_differences():
    data = fixture()
    for row in data['queries']:
        row['teacher_mrr_at_k'] = row['bm25_mrr_at_k']
        row['teacher_ndcg_at_k'] = row['bm25_ndcg_at_k']
        row['delta_mrr_at_k'] = row['delta_ndcg_at_k'] = 0
    report = paired_statistics(data, replicates=100)
    for metric in report['metrics'].values():
        assert metric['bootstrap_percentile_ci'] == [0, 0]
        assert metric['paired_sign_flip_p_two_sided'] == 1


@pytest.mark.parametrize('change', [
    lambda d: d.update(query_count=9),
    lambda d: d['queries'][1].update(query_id='0'),
    lambda d: d['queries'][0].update(teacher_mrr_at_k=float('nan')),
    lambda d: d['queries'][0].update(teacher_ndcg_at_k=1.1),
    lambda d: d['queries'][0].update(delta_mrr_at_k=0),
    lambda d: d['queries'][0].pop('bm25_mrr_at_k'),
    lambda d: d.update(queries=[]),
])
def test_invalid_data(change):
    d = fixture()
    change(d)
    with pytest.raises(ValueError):
        paired_statistics(d, replicates=100)


@pytest.mark.parametrize('kwargs', [
    {'replicates': 99}, {'replicates': 100.5}, {'seed': -1},
    {'confidence': 0}, {'confidence': 1}, {'confidence': float('nan')},
])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        paired_statistics(fixture(), **kwargs)


def test_intervals_and_p_values_in_range():
    report = paired_statistics(fixture(), replicates=300, seed=42, confidence=.9)
    for item in report['metrics'].values():
        lo, hi = item['bootstrap_percentile_ci']
        assert lo <= item['mean_teacher_minus_bm25'] <= hi
        assert 0 < item['paired_sign_flip_p_two_sided'] <= 1
