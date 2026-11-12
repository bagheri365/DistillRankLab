"""8B: full train-only SciFact top-100 comparison; never read official test data."""
import argparse
import json
from pathlib import Path
import time

from distillranklab.benchmark import bm25_order_scores, load_benchmark, summarize_ranking
from distillranklab.nonlinear_student import fit_nonlinear, predict_nonlinear
from distillranklab.real_data import split_groups
from distillranklab.real_student import fit_student, predict_student
from distillranklab.scaled_text import cached_embeddings_sqlite
from distillranklab.teacher import DEFAULT_MODEL, score_groups
from distillranklab.text_student import (fit_text_head, frozen_sentence_encoder,
                                         predict_text_head, text_feature_groups)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-data', type=Path, default=Path('datasets/scifact_full_train_bm25'))
    p.add_argument('--teacher-cache', type=Path, default=Path('datasets/scifact_full_train_teacher_scores.json'))
    p.add_argument('--embedding-cache', type=Path, default=Path('datasets/scifact_8b_embeddings.sqlite'))
    p.add_argument('--output', type=Path, default=Path('datasets/scifact_8b_comparison.json'))
    p.add_argument('--head-output', type=Path, default=Path('datasets/scifact_8b_head.json'))
    p.add_argument('--encoder', default='sentence-transformers/all-MiniLM-L6-v2')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--steps', type=int, default=120)
    p.add_argument('--allow-model-download', action='store_true')
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        p.error('steps and batch-size must be positive')
    groups, manifest = load_benchmark(args.train_data)
    if manifest.get('split') != 'train' or manifest.get('max_queries') is not None or len(groups) != 809:
        p.error('full official SciFact train benchmark (809 queries) required')
    if any(len(g.candidates) != 100 for g in groups):
        p.error('BM25 top-100 candidate pool required for every query')
    fit, val = split_groups(tuple(groups), test_fraction=0.2, seed=42)
    if len(fit) != 647 or len(val) != 162:
        p.error('unexpected split; verify query IDs and seed')

    def no_teacher_inference(_):
        raise RuntimeError('teacher cache incomplete: cross-encoder inference prohibited')

    teacher = score_groups(fit, scorer=no_teacher_inference, model_id=DEFAULT_MODEL,
                           cache_path=args.teacher_cache, batch_size=16)
    config = dict(mode='hybrid', feature_set='rich', temperature=2.0,
                  hard_weight=0.5, steps=args.steps, learning_rate=0.05, seed=42)
    results = {}
    for name, train, predict in (
        ('linear', lambda: fit_student(fit, teacher, **config), predict_student),
        ('mlp8', lambda: fit_nonlinear(fit, teacher, hidden_units=8, **config), predict_nonlinear),
    ):
        started = time.perf_counter()
        model = train()
        scores = predict(val, model)
        results[name] = {'metrics': summarize_ranking(val, scores, k=10),
                         'fit_and_validation_seconds': time.perf_counter() - started,
                         'parameters': 8 if name == 'linear' else 81}
        print(f'{name}: nDCG@10={results[name]["metrics"]["ndcg_at_k_all_queries"]:.6f}', flush=True)

    texts = list(dict.fromkeys(t for g in fit + val for t in
                               [g.query, *(c.passage for c in g.candidates)]))
    encoder = None

    def encode_missing(batch):
        nonlocal encoder
        if not args.allow_model_download:
            raise RuntimeError('embedding cache incomplete; rerun with --allow-model-download')
        if encoder is None:
            encoder = frozen_sentence_encoder(args.encoder, batch_size=args.batch_size)
        return encoder(batch)

    embeddings, cache_info = cached_embeddings_sqlite(
        texts, model_id=args.encoder, path=args.embedding_cache,
        encoder=encode_missing, batch_size=args.batch_size)
    started = time.perf_counter()
    fit_x = text_feature_groups(fit, embeddings)
    val_x = text_feature_groups(val, embeddings)
    feature_seconds = time.perf_counter() - started
    started = time.perf_counter()
    head = fit_text_head(fit, fit_x, teacher, steps=args.steps, learning_rate=0.05,
                         temperature=2.0, hard_weight=0.5, seed=42)
    train_seconds = time.perf_counter() - started
    started = time.perf_counter()
    scores = predict_text_head(val, val_x, head)
    validation_seconds = time.perf_counter() - started
    results['semantic_head'] = {'metrics': summarize_ranking(val, scores, k=10),
                                'parameters': 10, 'head_fit_seconds': train_seconds,
                                'head_validation_seconds': validation_seconds,
                                'feature_construction_seconds': feature_seconds}
    report = {'experiment': '8B scaled frozen text encoder', 'exploratory': True,
              'official_test_used': False, 'fit_queries': len(fit),
              'validation_queries': len(val), 'candidate_pool': 'BM25 top-100',
              'fit_candidates': sum(len(g.candidates) for g in fit),
              'validation_candidates': sum(len(g.candidates) for g in val),
              'encoder': args.encoder, 'encoder_frozen': True, 'steps': args.steps,
              'training_seed': 42, 'embedding_cache': cache_info,
              'bm25': summarize_ranking(val, bm25_order_scores(val), k=10),
              'models': results,
              'warning': 'Previously reused train-only validation split: exploratory; not independent confirmation.'}
    for path, data in ((args.output, report), (args.head_output, {**head, 'encoder': args.encoder})):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'report': str(args.output), 'head': str(args.head_output), **report}, indent=2))


if __name__ == '__main__':
    main()
