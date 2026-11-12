"""Descriptive, label-aware query error analysis on an unchanged candidate pool.

All categories are post-hoc diagnostics, never training features or selection rules.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import mean

from .benchmark import per_query_comparison
from .evaluation_audit import candidate_fingerprint, validate_score_pool
from .real_data import QueryGroup
from .real_student import TOKEN


def _ranked(group: QueryGroup, scores: Mapping) -> list:
    # Stable tie breaking follows original BM25 candidate order.
    return sorted(group.candidates, key=lambda c: -float(scores[group.query_id, c.passage_id]))


def _attributes(group: QueryGroup, bm25: Mapping, semantic: Mapping, combined: Mapping) -> dict:
    qterms = set(TOKEN.findall(group.query.lower()))
    first = group.candidates[0]
    top_terms = set(TOKEN.findall(first.passage.lower()))
    overlap = len(qterms & top_terms) / max(1, len(qterms))
    positives = [i + 1 for i, c in enumerate(group.candidates)
                 if c.relevance is not None and c.relevance > 0]
    return {
        'query_tokens': len(TOKEN.findall(group.query.lower())),
        'bm25_top1_query_token_overlap': overlap,
        'best_relevant_bm25_rank': min(positives) if positives else None,
        'judged_candidates': sum(c.relevance is not None for c in group.candidates),
        'unjudged_candidates': sum(c.relevance is None for c in group.candidates),
        'bm25_top1_id': _ranked(group, bm25)[0].passage_id,
        'semantic_top1_id': _ranked(group, semantic)[0].passage_id,
        'combined_top1_id': _ranked(group, combined)[0].passage_id,
    }


def _category_rows(rows: list[dict], key: str, categories: Sequence[str]) -> dict:
    result = {}
    for category in categories:
        subset = [r for r in rows if r[key] == category]
        if not subset:
            result[category] = {'queries': 0, 'mean_delta_ndcg': None, 'improved': 0, 'regressed': 0}
            continue
        result[category] = {
            'queries': len(subset),
            'mean_delta_ndcg': mean(r['delta_ndcg_at_10'] for r in subset),
            'improved': sum(r['delta_ndcg_at_10'] > 1e-12 for r in subset),
            'regressed': sum(r['delta_ndcg_at_10'] < -1e-12 for r in subset),
        }
    return result


def query_error_report(groups: Sequence[QueryGroup], scores: Mapping[str, Mapping],
                       *, examples: int = 5) -> dict:
    """Produce query-level gains, losses, and descriptive stratified summaries.

    Requires exactly bm25, semantic_only, combined scores for the same pool.
    No unjudged candidate is silently treated as a judged negative in metadata.
    """
    if not isinstance(examples, int) or isinstance(examples, bool) or examples < 0 or examples > 50:
        raise ValueError('examples must be an integer between 0 and 50')
    if set(scores) != {'bm25', 'semantic_only', 'combined'}:
        raise ValueError('required rankers: bm25, semantic_only, combined')
    for name, values in scores.items():
        validate_score_pool(groups, values, name)
    attributes = {g.query_id: _attributes(g, scores['bm25'], scores['semantic_only'], scores['combined'])
                  for g in groups}
    comparisons = {}
    for baseline in ('bm25', 'combined'):
        diagnostics = per_query_comparison(groups, scores[baseline], scores['semantic_only'], k=10)
        rows = []
        for record in diagnostics['queries']:
            attrs = attributes[record['query_id']]
            length = attrs['query_tokens']
            overlap = attrs['bm25_top1_query_token_overlap']
            rank = attrs['best_relevant_bm25_rank']
            row = {
                'query_id': record['query_id'],
                'retrieved_positive': record['retrieved_positive'],
                'baseline_ndcg_at_10': record['bm25_ndcg_at_k'],
                'semantic_ndcg_at_10': record['teacher_ndcg_at_k'],
                'delta_ndcg_at_10': record['delta_ndcg_at_k'],
                'delta_mrr_at_10': record['delta_mrr_at_k'],
                'query_length_bin': 'short_1_5' if length <= 5 else 'medium_6_10' if length <= 10 else 'long_11_plus',
                'overlap_bin': 'low_below_0_25' if overlap < .25 else 'mid_0_25_to_0_5' if overlap < .5 else 'high_0_5_plus',
                'positive_rank_bin': 'not_retrieved' if rank is None else 'top_10' if rank <= 10 else 'rank_11_100',
                **attrs,
            }
            rows.append(row)
        descending = sorted(rows, key=lambda r: (-r['delta_ndcg_at_10'], r['query_id']))
        ascending = sorted(rows, key=lambda r: (r['delta_ndcg_at_10'], r['query_id']))
        comparisons[f'semantic_only_vs_{baseline}'] = {
            'baseline': baseline, 'challenger': 'semantic_only',
            'queries': len(rows), 'mean_delta_ndcg': mean(r['delta_ndcg_at_10'] for r in rows),
            'improved': sum(r['delta_ndcg_at_10'] > 1e-12 for r in rows),
            'regressed': sum(r['delta_ndcg_at_10'] < -1e-12 for r in rows),
            'unchanged': sum(abs(r['delta_ndcg_at_10']) <= 1e-12 for r in rows),
            'top_gains': [r for r in descending if r['delta_ndcg_at_10'] > 1e-12][:examples],
            'top_losses': [r for r in ascending if r['delta_ndcg_at_10'] < -1e-12][:examples],
            'by_query_length': _category_rows(rows, 'query_length_bin', ('short_1_5', 'medium_6_10', 'long_11_plus')),
            'by_top1_overlap': _category_rows(rows, 'overlap_bin', ('low_below_0_25', 'mid_0_25_to_0_5', 'high_0_5_plus')),
            'by_positive_bm25_rank': _category_rows(rows, 'positive_rank_bin', ('not_retrieved', 'top_10', 'rank_11_100')),
            'per_query': rows,
        }
    return {
        'experiment': '8E query-level semantic error analysis',
        'exploratory': True, 'official_test_used': False,
        'candidate_pool_sha256': candidate_fingerprint(groups),
        'comparisons': comparisons,
        'limitations': ('Post-hoc descriptive bins; sparse qrels and unjudged-as-nonrelevant metrics; '
                        'no causal claims or independent validation; original BM25 ranks are candidate order.'),
    }
