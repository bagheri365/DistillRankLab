"""10A rank blending and conservative guard are label-independent."""
import numpy as np
import pytest
from distillranklab.real_data import Candidate, QueryGroup
from distillranklab.conservative_reranking import rank_scores, reranking_suite


def fixture():
    g = QueryGroup(query_id='q', query='question', candidates=tuple(
        Candidate(passage_id=str(i), passage=f'doc {i}', relevance=i % 2)
        for i in range(4)))
    x = np.zeros((4, 10))
    x[:, 8] = [.70, .75, .80, .90]
    return [g], [x]


def ordered(scores):
    return sorted(range(4), key=lambda i: -scores['q', str(i)])


def test_baselines():
    g, x = fixture()
    assert ordered(rank_scores(g, x, method='bm25')) == [0, 1, 2, 3]
    assert ordered(rank_scores(g, x, method='cosine')) == [3, 2, 1, 0]


def test_blend_endpoints():
    g, x = fixture()
    assert ordered(rank_scores(g, x, method='blend', alpha=0)) == [0, 1, 2, 3]
    assert ordered(rank_scores(g, x, method='blend', alpha=1)) == [3, 2, 1, 0]


def test_guard_preserves_bm25_top1_only_for_small_gap():
    g, x = fixture()
    # Large gap: no protection.
    assert ordered(rank_scores(g, x, method='guarded_blend', alpha=1, guard_gap=.05))[0] == 3
    # Small gap: BM25 top1 protected.
    assert ordered(rank_scores(g, x, method='guarded_blend', alpha=1, guard_gap=.25))[0] == 0


def test_guard_does_not_change_non_top1_relative_order():
    g, x = fixture()
    ranks = ordered(rank_scores(g, x, method='guarded_blend', alpha=1, guard_gap=.25))
    assert ranks == [0, 3, 2, 1]


def test_label_independence():
    g, x = fixture()
    unlabeled = [QueryGroup(query_id=g[0].query_id, query=g[0].query,
                           candidates=tuple(Candidate(passage_id=c.passage_id, passage=c.passage,
                                                      relevance=None) for c in g[0].candidates))]
    for method in ('bm25', 'cosine', 'blend', 'guarded_blend'):
        assert rank_scores(g, x, method=method) == rank_scores(unlabeled, x, method=method)


def test_suite_predeclared():
    g, x = fixture()
    scores = reranking_suite(g, x)
    assert set(scores) == {'bm25', 'cosine', 'blend_0.25', 'blend_0.5', 'blend_0.75',
                           'guarded_0.25', 'guarded_0.5', 'guarded_0.75'}
    assert all(len(v) == 4 for v in scores.values())


@pytest.mark.parametrize('method,alpha,gap', [('invalid', .5, .05), ('blend', -1, .05),
                                               ('blend', 2, .05), ('blend', .5, -1),
                                               ('blend', float('nan'), .05)])
def test_bad_config(method, alpha, gap):
    g, x = fixture()
    with pytest.raises(ValueError):
        rank_scores(g, x, method=method, alpha=alpha, guard_gap=gap)


def test_bad_features_and_duplicate_ids():
    g, x = fixture()
    with pytest.raises(ValueError):
        rank_scores(g, [np.zeros((4, 9))], method='cosine')
    with pytest.raises(ValueError):
        rank_scores(g, [np.full((4, 10), np.nan)], method='cosine')
    with pytest.raises(ValueError):
        reranking_suite(g, x, alphas=[.5, .5])
    with pytest.raises(ValueError):
        reranking_suite(g, x, alphas=[0])
