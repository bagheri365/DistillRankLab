"""8E: post-hoc query error analysis on SciFact train-only validation pool."""
import argparse
import json
from pathlib import Path

from distillranklab.benchmark import bm25_order_scores, load_benchmark
from distillranklab.evaluation_audit import candidate_fingerprint
from distillranklab.query_errors import query_error_report
from distillranklab.real_data import split_groups
from distillranklab.scaled_text import cached_embeddings_sqlite
from distillranklab.semantic_ablation import run_feature_ablation
from distillranklab.teacher import DEFAULT_MODEL, score_groups
from distillranklab.text_student import text_feature_groups

EXPECTED_VALIDATION_SHA256 = 'efca1a9fa6463a831749c747b9a3d1543f18161c474aadec63750949aa6cc440'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-data', type=Path, default=Path('datasets/scifact_full_train_bm25'))
    p.add_argument('--teacher-cache', type=Path, default=Path('datasets/scifact_full_train_teacher_scores.json'))
    p.add_argument('--embedding-cache', type=Path, default=Path('datasets/scifact_8b_embeddings.sqlite'))
    p.add_argument('--output', type=Path, default=Path('datasets/scifact_8e_query_errors.json'))
    p.add_argument('--encoder', default='sentence-transformers/all-MiniLM-L6-v2')
    p.add_argument('--examples', type=int, default=5)
    args = p.parse_args()
    groups, manifest = load_benchmark(args.train_data)
    if (manifest.get('split') != 'train' or manifest.get('max_queries') is not None
            or len(groups) != 809 or any(len(g.candidates) != 100 for g in groups)):
        p.error('full official SciFact TRAIN top-100 required; official test prohibited')
    fit, val = split_groups(tuple(groups), test_fraction=.2, seed=42)
    if len(fit) != 647 or len(val) != 162 or candidate_fingerprint(val) != EXPECTED_VALIDATION_SHA256:
        p.error('validation candidate pool fingerprint mismatch against 8D')

    def forbidden(_):
        raise RuntimeError('incomplete cache: new inference prohibited')

    teacher = score_groups(fit, scorer=forbidden, model_id=DEFAULT_MODEL,
                           cache_path=args.teacher_cache, batch_size=16)
    texts = list(dict.fromkeys(t for g in fit + val for t in [g.query, *(c.passage for c in g.candidates)]))
    embeddings, cache = cached_embeddings_sqlite(texts, model_id=args.encoder,
                                                  path=args.embedding_cache, encoder=forbidden, log=None)
    if cache['new_embeddings']:
        raise RuntimeError('new embeddings prohibited')
    _, predictions = run_feature_ablation(fit, val, text_feature_groups(fit, embeddings),
                                         text_feature_groups(val, embeddings), teacher)
    report = query_error_report(val, {'bm25': bm25_order_scores(val),
                                      'semantic_only': predictions['semantic_only'],
                                      'combined': predictions['combined']}, examples=args.examples)
    report['embedding_cache'] = cache
    report['fit_queries'] = len(fit)
    report['validation_queries'] = len(val)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'report': str(args.output), 'experiment': report['experiment'],
                      'official_test_used': False, 'validation_pool_sha256': report['candidate_pool_sha256'],
                      'comparisons': {name: {key: value[key] for key in
                                            ('queries', 'mean_delta_ndcg', 'improved', 'regressed', 'unchanged',
                                             'top_gains', 'top_losses', 'by_query_length', 'by_top1_overlap',
                                             'by_positive_bm25_rank')}
                                      for name, value in report['comparisons'].items()}}, indent=2))


if __name__ == '__main__':
    main()
