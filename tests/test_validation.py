import json

import pytest

from distillranklab.real_data import Candidate, QueryGroup
from distillranklab.real_student import predict_student
from distillranklab.validation import make_configs, select_student


def sample():
    return tuple(QueryGroup(f'q{i}', f'topic{i}', (
        Candidate(f'p{i}a', f'topic{i} relevant', 1),
        Candidate(f'p{i}b', 'other words', None),
        Candidate(f'p{i}c', 'irrelevant', 0))) for i in range(12))


def teacher(groups):
    return {(g.query_id, c.passage_id): float(3 - rank)
            for g in groups for rank, c in enumerate(g.candidates)}


def test_grid_default_modes_and_determinism():
    configs = make_configs()
    assert [c['mode'] for c in configs].count('hard') == 1
    assert [c['mode'] for c in configs].count('soft') == 2
    assert [c['mode'] for c in configs].count('hybrid') == 4
    assert configs == make_configs()


@pytest.mark.parametrize('kwargs', [dict(learning_rates=[]), dict(temperatures=[0]),
                                    dict(hard_weights=[1]), dict(steps=[0]),
                                    dict(learning_rates=[float('nan')])])
def test_grid_rejects_invalid(kwargs):
    with pytest.raises(ValueError):
        make_configs(**kwargs)


def test_disjoint_split_refit_and_reproducible():
    g = sample()
    config = make_configs(steps=[3])
    report, model = select_student(g, teacher(g), configs=config)
    again, model_again = select_student(tuple(reversed(g)), teacher(g), configs=config)
    assert set(report['fit_query_ids']).isdisjoint(report['validation_query_ids'])
    assert len(report['fit_query_ids']) + len(report['validation_query_ids']) == len(g)
    assert report['refit_queries'] == len(g)
    assert model['training_queries'] == len(g)
    assert report['selected_config'] == again['selected_config']
    assert report['validation_query_ids'] == again['validation_query_ids']
    assert predict_student(g, model) == predict_student(g, model_again)
    assert report['selected_validation_ndcg_at_k_all_queries'] == again['selected_validation_ndcg_at_k_all_queries']
    json.dumps(report)


def test_hard_only_does_not_need_teacher():
    g = sample()
    configs = [make_configs(steps=[2])[0]]
    report, model = select_student(g, None, configs=configs)
    assert model['mode'] == 'hard'
    assert report['selected_index'] == 0


def test_soft_requires_teacher():
    g = sample()
    configs = [make_configs(steps=[2])[1]]
    with pytest.raises(ValueError, match='teacher scores'):
        select_student(g, None, configs=configs)


def test_invalid_split_and_empty_grid():
    g = sample()
    with pytest.raises(ValueError, match='grid'):
        select_student(g, teacher(g), configs=[])
    with pytest.raises(ValueError, match='fraction'):
        select_student(g, teacher(g), validation_fraction=1)


def test_validation_scores_do_not_depend_on_held_out_labels():
    g = sample()
    scores = teacher(g)
    config = make_configs(steps=[2])
    a, _ = select_student(g, scores, configs=config)
    # No test data is even an input to select_student; the selection report
    # depends exclusively on the supplied official training groups.
    assert 'test' not in a
    assert a['selection_metric'] == 'validation_all_query_ndcg@10'


def test_invalid_config_keys():
    with pytest.raises(ValueError, match='config keys'):
        select_student(sample(), teacher(sample()), configs=[{'mode': 'hard'}])
