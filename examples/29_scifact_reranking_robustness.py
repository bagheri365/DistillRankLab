"""10B: frozen-shortlist robustness audit on reused SciFact validation.

No model inference, official test, training, or hyperparameter tuning.
"""
import argparse
import json
from pathlib import Path

from distillranklab.benchmark import load_benchmark
from distillranklab.evaluation_audit import candidate_fingerprint
from distillranklab.real_data import split_groups
from distillranklab.reranking_robustness import audit_pool, shortlist_audit
from distillranklab.scaled_text import cached_embeddings_sqlite
from distillranklab.text_student import text_feature_groups

EXPECTED_SHA = 'efca1a9fa6463a831749c747b9a3d1543f18161c474aadec63750949aa6cc440'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-data', type=Path, default=Path('datasets/scifact_full_train_bm25'))
    parser.add_argument('--embedding-cache', type=Path, default=Path('datasets/scifact_8b_embeddings.sqlite'))
    parser.add_argument('--output', type=Path, default=Path('datasets/scifact_10b_reranking_robustness.json'))
    parser.add_argument('--encoder', default='sentence-transformers/all-MiniLM-L6-v2')
    args = parser.parse_args()
    groups, manifest = load_benchmark(args.train_data)
    if (manifest.get('split') != 'train' or manifest.get('max_queries') is not None
            or len(groups) != 809 or any(len(g.candidates) != 100 for g in groups)):
        parser.error('full SciFact TRAIN BM25 top-100 required')
    fit, val = split_groups(tuple(groups), test_fraction=.2, seed=42)
    if len(fit) != 647 or len(val) != 162 or candidate_fingerprint(val) != EXPECTED_SHA:
        parser.error('validation pool differs from audited fingerprint')

    def forbidden(_):
        raise RuntimeError('embedding cache incomplete: new inference prohibited')

    texts = list(dict.fromkeys(t for g in val for t in [g.query, *(c.passage for c in g.candidates)]))
    embeddings, cache = cached_embeddings_sqlite(
        texts, model_id=args.encoder, path=args.embedding_cache, encoder=forbidden, log=None)
    if cache['new_embeddings']:
        raise RuntimeError('new embeddings prohibited')
    matrices = text_feature_groups(val, embeddings)
    audit = shortlist_audit(val, matrices)
    pools = audit_pool(val, matrices)
    report = {
        'experiment': '10B frozen-shortlist reranking robustness',
        'exploratory': True, 'official_test_used': False,
        'fit_queries_not_trained': len(fit), 'validation_queries': len(val),
        'validation_pool_sha256': EXPECTED_SHA,
        'shortlist': ['bm25', 'cosine', 'blend_0.5', 'blend_0.75'],
        'encoder': args.encoder, 'encoder_frozen': True, 'cache': cache,
        'full_pool_audit': audit, 'bm25_pool_truncation': pools,
        'limitations': [
            'Repeatedly reused validation: descriptive audit only, not confirmatory',
            'Shortlist chosen after observing 10A validation: selection bias',
            'No fresh model inference or end-to-end CPU latency measurement',
            'Truncation changes candidate coverage and ranking opportunity',
            'Unjudged candidates treated as nonrelevant in metrics',
            'Qrel-based rank transitions are post-hoc diagnostics, never inference-time routing',
            'No distillation mechanism in these rankers',
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    summary = {'report': str(args.output), 'experiment': report['experiment'],
               'official_test_used': False, 'validation_queries': len(val),
               'validation_pool_sha256': EXPECTED_SHA,
               'models': audit['models'], 'top1_agreement': audit['top1_agreement'],
               'top10_overlap_mean': audit['top10_overlap_mean'],
               'comparisons_vs_bm25': {name: {k: v for k, v in item.items() if k != 'per_query'}
                                       for name, item in audit['comparisons_vs_bm25'].items()},
               'comparisons_vs_cosine': {name: {k: v for k, v in item.items() if k != 'per_query'}
                                        for name, item in audit['comparisons_vs_cosine'].items()},
               'positive_rank_transition_counts': audit['positive_rank_transitions_bm25_to_blend_0.75']['counts'],
               'truncation_metrics': {n: d['models'] for n, d in pools.items()},
               'limitations': report['limitations']}
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
