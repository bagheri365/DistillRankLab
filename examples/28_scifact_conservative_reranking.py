"""10A: predeclared BM25/semantic blending on exploratory SciFact validation.

Uses cached frozen-encoder embeddings only; no teacher or official test access.
"""
import argparse
import json
from pathlib import Path

from distillranklab.benchmark import load_benchmark, summarize_ranking, per_query_comparison
from distillranklab.conservative_reranking import reranking_suite
from distillranklab.evaluation_audit import candidate_fingerprint, validate_score_pool
from distillranklab.real_data import split_groups
from distillranklab.scaled_text import cached_embeddings_sqlite
from distillranklab.semantic_ablation import compare_paired
from distillranklab.text_student import text_feature_groups

EXPECTED_SHA = 'efca1a9fa6463a831749c747b9a3d1543f18161c474aadec63750949aa6cc440'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-data', type=Path, default=Path('datasets/scifact_full_train_bm25'))
    parser.add_argument('--embedding-cache', type=Path, default=Path('datasets/scifact_8b_embeddings.sqlite'))
    parser.add_argument('--output', type=Path, default=Path('datasets/scifact_10a_conservative_reranking.json'))
    parser.add_argument('--encoder', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--replicates', type=int, default=10000)
    args = parser.parse_args()
    if args.replicates < 100:
        parser.error('replicates must be >= 100')
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
    embeddings, cache = cached_embeddings_sqlite(texts, model_id=args.encoder,
                                                  path=args.embedding_cache, encoder=forbidden, log=None)
    if cache['new_embeddings']:
        raise RuntimeError('new embeddings prohibited')
    matrices = text_feature_groups(val, embeddings)
    scores = reranking_suite(val, matrices)
    for name, mapping in scores.items():
        validate_score_pool(val, mapping, name)
    metrics = {name: summarize_ranking(val, mapping, k=10) for name, mapping in scores.items()}
    comparisons = {f'{name}_vs_bm25': compare_paired(val, scores['bm25'], mapping,
                   baseline_name='bm25', challenger_name=name, replicates=args.replicates)
                   for name, mapping in scores.items() if name != 'bm25'}
    cosine_comparisons = {f'{name}_vs_cosine': compare_paired(val, scores['cosine'], scores[name],
                          baseline_name='cosine', challenger_name=name, replicates=args.replicates)
                          for name in scores if name.startswith(('blend_', 'guarded_'))}
    outcomes = {name: {key: per_query_comparison(val, scores[key], scores[name], k=10)[key2]
                        for key, key2 in [('bm25', 'ndcg_regressed_queries'),
                                          ('cosine', 'ndcg_regressed_queries')]}
                for name in scores if name.startswith(('blend_', 'guarded_'))}
    report = {'experiment': '10A conservative semantic reranking', 'exploratory': True,
              'official_test_used': False, 'fit_queries_not_trained': len(fit),
              'validation_queries': len(val), 'validation_pool_sha256': EXPECTED_SHA,
              'candidate_pool': 'BM25 top-100', 'encoder': args.encoder,
              'encoder_frozen': True, 'cache': cache, 'replicates': args.replicates,
              'rules': {'bm25': 'original candidate order, not numerical BM25 scores',
                        'cosine': 'frozen embedding cosine',
                        'blend': 'convex combination of normalized BM25 rank and cosine rank',
                        'alphas': [.25, .5, .75],
                        'guarded': 'preserve BM25 top1 if cosine winner advantage <= 0.05',
                        'guard_gap': .05, 'uses_qrels_or_teacher_for_routing': False},
              'models': metrics, 'paired_vs_bm25': comparisons,
              'paired_vs_cosine': cosine_comparisons,
              'regressed_queries_vs_baselines': outcomes,
              'limitations': ['Validation reused repeatedly; no confirmatory inference',
                              'Predeclared grid is exploratory, not independently tuned',
                              'Unjudged passages counted nonrelevant',
                              'Rank-prior blending uses BM25 order, not raw BM25 scores',
                              'Cached feature construction is not end-to-end inference latency',
                              'No distillation is used in these reranking rules',
                              'Multiple unadjusted paired comparisons']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'report': str(args.output), **report}, indent=2))


if __name__ == '__main__':
    main()
