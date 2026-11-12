import math

import pytest

from distillranklab.evaluation_audit import audit_pair, candidate_fingerprint, validate_score_pool
from distillranklab.real_data import Candidate, QueryGroup


def sample():
    groups = (
        QueryGroup('q2', 'second', (Candidate('p3', 'x', 1), Candidate('p4', 'y', None))),
        QueryGroup('q1', 'first', (Candidate('p1', 'x', 1), Candidate('p2', 'y', None))),
    )
    base = {('q1', 'p1'): 2., ('q1', 'p2'): 1., ('q2', 'p3'): 1., ('q2', 'p4'): 2.}
    challenge = {('q1', 'p1'): 1., ('q1', 'p2'): 2., ('q2', 'p3'): 2., ('q2', 'p4'): 1.}
    return groups, base, challenge


def test_audit_aggregates_both_policies():
    groups, base, challenger = sample()
    report = audit_pair(groups, base, challenger, baseline_name='base', challenger_name='challenger')
    assert report['pool']['candidate_count'] == 4
    assert report['policies']['nonrelevant']['per_query_aggregate_verified']
    assert report['policies']['judged_only']['per_query_aggregate_verified']
    assert report['policies']['nonrelevant']['delta_distribution']['mrr']['mean_delta'] == 0.0
    assert report['policies']['judged_only']['delta_distribution']['mrr']['mean_delta'] == 0.0
    assert report['policies']['nonrelevant']['delta_distribution']['ndcg']['improved_queries'] == 1
    assert report['policies']['nonrelevant']['delta_distribution']['ndcg']['regressed_queries'] == 1
    assert report['policies']['judged_only']['delta_distribution']['ndcg']['unchanged_queries'] == 2


def test_fingerprint_query_order_independent():
    groups, _, _ = sample()
    assert candidate_fingerprint(groups) == candidate_fingerprint(groups[::-1])


def test_fingerprint_sensitive_to_candidate_order():
    groups, _, _ = sample()
    changed = (QueryGroup(groups[0].query_id, groups[0].query, groups[0].candidates[::-1]), groups[1])
    assert candidate_fingerprint(groups) != candidate_fingerprint(changed)


def test_fingerprint_sensitive_to_judgments():
    groups, _, _ = sample()
    changed = (QueryGroup('q2', 'second', (Candidate('p3', 'x', 0), Candidate('p4', 'y', None))), groups[1])
    assert candidate_fingerprint(groups) != candidate_fingerprint(changed)


def test_missing_score_rejected():
    groups, base, _ = sample()
    base.pop(('q1', 'p1'))
    with pytest.raises(ValueError, match='pool mismatch'):
        validate_score_pool(groups, base, 'bad')


def test_extra_score_rejected():
    groups, base, _ = sample()
    base[('q0', 'p0')] = 1.
    with pytest.raises(ValueError, match='pool mismatch'):
        validate_score_pool(groups, base, 'bad')


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), '1', True])
def test_bad_scores_rejected(bad):
    groups, base, _ = sample()
    base[('q1', 'p1')] = bad
    with pytest.raises(ValueError, match='score'):
        validate_score_pool(groups, base, 'bad')


def test_duplicate_query_rejected():
    groups, _, _ = sample()
    with pytest.raises(ValueError, match='duplicate query'):
        candidate_fingerprint(groups + (groups[0],))


def test_duplicate_candidate_rejected():
    groups, _, _ = sample()
    broken = (QueryGroup('q', 'x', (Candidate('p', 'x', 1), Candidate('p', 'x', None))),)
    with pytest.raises(ValueError, match='duplicate candidate'):
        candidate_fingerprint(broken)


def test_empty_rejected():
    with pytest.raises(ValueError, match='empty candidate'):
        candidate_fingerprint(())


def test_nonrelevant_and_judged_only_differ():
    groups, base, challenger = sample()
    report = audit_pair(groups, base, challenger, baseline_name='base', challenger_name='challenger')
    p = report['policies']
    assert p['nonrelevant']['baseline']['mrr_at_k_all_queries'] < p['judged_only']['baseline']['mrr_at_k_all_queries']
