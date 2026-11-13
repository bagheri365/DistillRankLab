"""9B: inspect actual head learning, teacher agreement and ranking changes.

Cache-only exploratory SciFact TRAIN experiment. Does not touch official test.
"""
import argparse
import json
from pathlib import Path

from distillranklab.benchmark import load_benchmark
from distillranklab.evaluation_audit import candidate_fingerprint
from distillranklab.learning_diagnostics import learning_diagnostics
from distillranklab.real_data import split_groups
from distillranklab.scaled_text import cached_embeddings_sqlite
from distillranklab.teacher import DEFAULT_MODEL, score_groups
from distillranklab.text_student import text_feature_groups

EXPECTED_SHA = 'efca1a9fa6463a831749c747b9a3d1543f18161c474aadec63750949aa6cc440'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-data', type=Path, default=Path('datasets/scifact_full_train_bm25'))
    parser.add_argument('--teacher-cache', type=Path, default=Path('datasets/scifact_full_train_teacher_scores.json'))
    parser.add_argument('--embedding-cache', type=Path, default=Path('datasets/scifact_8b_embeddings.sqlite'))
    parser.add_argument('--output', type=Path, default=Path('datasets/scifact_9b_learning_diagnostics.json'))
    parser.add_argument('--encoder', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--steps', type=int, default=120)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('steps must be positive')
    groups, manifest = load_benchmark(args.train_data)
    if (manifest.get('split') != 'train' or manifest.get('max_queries') is not None
            or len(groups) != 809 or any(len(g.candidates) != 100 for g in groups)):
        parser.error('full official SciFact TRAIN BM25 top-100 required')
    fit, val = split_groups(tuple(groups), test_fraction=.2, seed=42)
    if len(fit) != 647 or len(val) != 162 or candidate_fingerprint(val) != EXPECTED_SHA:
        parser.error('validation pool differs from audited 8D fingerprint')

    def forbidden(_):
        raise RuntimeError('cache incomplete: model inference prohibited')

    teacher = score_groups(fit, scorer=forbidden, model_id=DEFAULT_MODEL,
                           cache_path=args.teacher_cache, batch_size=16)
    texts = list(dict.fromkeys(t for g in fit + val for t in [g.query, *(c.passage for c in g.candidates)]))
    embeddings, cache = cached_embeddings_sqlite(texts, model_id=args.encoder,
                                                  path=args.embedding_cache, encoder=forbidden, log=None)
    if cache['new_embeddings']:
        raise RuntimeError('new embeddings prohibited')
    fit_x = text_feature_groups(fit, embeddings)
    val_x = text_feature_groups(val, embeddings)
    results = learning_diagnostics(fit, val, fit_x, val_x, teacher, steps=args.steps)
    report = {'experiment': '9B distillation learning diagnostics', 'exploratory': True,
              'official_test_used': False, 'fit_queries': len(fit), 'validation_queries': len(val),
              'validation_pool_sha256': EXPECTED_SHA, 'cache': cache, 'steps': args.steps,
              'seed': 42, 'feature_blocks': results,
              'limitations': ['Fit teacher agreement is in-sample and not a generalization metric',
                              'Same repeatedly reused validation pool',
                              'Sparse judgments; no inference latency measured',
                              'Losses are diagnostic objectives, not ranking quality']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'report': str(args.output), 'experiment': report['experiment'],
                      'official_test_used': False, 'fit_queries': len(fit),
                      'validation_queries': len(val), 'cache': cache,
                      'feature_blocks': {block: {name: {
                          'soft_kl_initial': values['trajectory'][0]['soft_kl'],
                          'soft_kl_final': values['trajectory'][-1]['soft_kl'],
                          'hard_ce_initial': values['trajectory'][0]['hard_cross_entropy'],
                          'hard_ce_final': values['trajectory'][-1]['hard_cross_entropy'],
                          'validation_order_vs_initial': values['validation_order_vs_initial'],
                          'fit_teacher_agreement': values['fit_teacher_agreement'],
                          'active_weights': values['active_weights']}
                          for name, values in models.items()}
                          for block, models in results.items()}}, indent=2))


if __name__ == '__main__':
    main()
