"""9D: label-free confidence-weighted hybrid distillation with shuffled control.

Train-only exploratory validation. Frozen embeddings and teacher scores must
already be cached. Never accesses official SciFact test data.
"""
import argparse
import json
from pathlib import Path

from distillranklab.benchmark import bm25_order_scores, load_benchmark, summarize_ranking
from distillranklab.evaluation_audit import candidate_fingerprint, validate_score_pool
from distillranklab.real_data import split_groups
from distillranklab.scaled_text import cached_embeddings_sqlite
from distillranklab.semantic_ablation import compare_paired
from distillranklab.confidence_distillation import confidence_ablation
from distillranklab.teacher import DEFAULT_MODEL, score_groups
from distillranklab.text_student import text_feature_groups

EXPECTED_SHA = 'efca1a9fa6463a831749c747b9a3d1543f18161c474aadec63750949aa6cc440'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-data', type=Path, default=Path('datasets/scifact_full_train_bm25'))
    parser.add_argument('--teacher-cache', type=Path, default=Path('datasets/scifact_full_train_teacher_scores.json'))
    parser.add_argument('--embedding-cache', type=Path, default=Path('datasets/scifact_8b_embeddings.sqlite'))
    parser.add_argument('--output', type=Path, default=Path('datasets/scifact_9d_confidence_distillation.json'))
    parser.add_argument('--encoder', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--replicates', type=int, default=10000)
    args = parser.parse_args()
    if args.steps < 1 or args.replicates < 100:
        parser.error('steps must be positive and replicates >= 100')
    groups, manifest = load_benchmark(args.train_data)
    if (manifest.get('split') != 'train' or manifest.get('max_queries') is not None
            or len(groups) != 809 or any(len(g.candidates) != 100 for g in groups)):
        parser.error('full official SciFact TRAIN BM25 top-100 required')
    fit, val = split_groups(tuple(groups), test_fraction=.2, seed=42)
    if len(fit) != 647 or len(val) != 162 or candidate_fingerprint(val) != EXPECTED_SHA:
        parser.error('validation pool differs from audited 8D fingerprint')

    def forbidden(_):
        raise RuntimeError('cache incomplete: new model inference prohibited')

    teacher = score_groups(fit, scorer=forbidden, model_id=DEFAULT_MODEL,
                           cache_path=args.teacher_cache, batch_size=16)
    texts = list(dict.fromkeys(t for g in fit + val for t in [g.query, *(c.passage for c in g.candidates)]))
    embeddings, cache = cached_embeddings_sqlite(texts, model_id=args.encoder,
                                                  path=args.embedding_cache, encoder=forbidden, log=None)
    if cache['new_embeddings']:
        raise RuntimeError('new embeddings prohibited')
    train_x = text_feature_groups(fit, embeddings)
    val_x = text_feature_groups(val, embeddings)
    bm25 = bm25_order_scores(val)
    validate_score_pool(val, bm25, 'bm25')
    metrics, scores = confidence_ablation(fit, val, train_x, val_x, teacher, steps=args.steps)
    for name, values in scores.items():
        validate_score_pool(val, values, name)
    comparisons = {f'{name}_vs_uniform': compare_paired(
        val, scores['uniform'], scores[name], baseline_name='uniform',
        challenger_name=name, replicates=args.replicates)
        for name in ('margin', 'agreement', 'shuffled_margin')}
    report = {'experiment': '9D confidence-aware distillation ablation',
              'exploratory': True, 'official_test_used': False,
              'fit_queries': len(fit), 'validation_queries': len(val),
              'validation_pool_sha256': EXPECTED_SHA, 'candidate_pool': 'BM25 top-100',
              'encoder': args.encoder, 'encoder_frozen': True, 'cache': cache,
              'steps': args.steps, 'seed': 42, 'replicates': args.replicates,
              'hard_weight': .5, 'temperature': 2.,
              'weight_rules': {'uniform': 'all queries equal',
                               'margin': '0.5 + top1-top2 gap/(1+gap), normalized',
                               'agreement': '1.5 if teacher top1 equals BM25 top1 else 0.5, normalized',
                               'shuffled_margin': 'margin weights permuted across queries with seed 2026'},
              'bm25': summarize_ranking(val, bm25, k=10),
              'models': metrics, 'paired_comparisons': comparisons,
              'limitations': ['Repeatedly reused exploratory validation; no confirmatory claims',
                              'Unjudged candidates treated as nonrelevant',
                              'Unadjusted multiple comparisons',
                              'Teacher agreement and margin do not establish correctness',
                              'One deterministic shuffle is a negative control, not a null distribution',
                              'Frozen pretrained encoder cost not benchmarked']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'report': str(args.output), **report}, indent=2))


if __name__ == '__main__':
    main()
