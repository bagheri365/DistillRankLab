"""8A small train-only frozen text-encoder feasibility study; no test split."""
import argparse
import json
from pathlib import Path
import time

from distillranklab.benchmark import bm25_order_scores, load_benchmark, summarize_ranking
from distillranklab.real_data import QueryGroup, split_groups
from distillranklab.teacher import DEFAULT_MODEL, score_groups
from distillranklab.text_student import (encode_cached, fit_text_head, frozen_sentence_encoder,
                                         predict_text_head, text_feature_groups)


def subset(groups, query_limit, candidate_limit):
    return tuple(QueryGroup(g.query_id, g.query, g.candidates[:candidate_limit])
                 for g in groups[:query_limit])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-data', type=Path, default=Path('datasets/scifact_full_train_bm25'))
    p.add_argument('--teacher-cache', type=Path, default=Path('datasets/scifact_full_train_teacher_scores.json'))
    p.add_argument('--embedding-cache', type=Path, default=Path('datasets/scifact_8a_embeddings.json'))
    p.add_argument('--output', type=Path, default=Path('datasets/scifact_8a_feasibility.json'))
    p.add_argument('--model-output', type=Path, default=Path('datasets/scifact_8a_head.json'))
    p.add_argument('--encoder', default='sentence-transformers/all-MiniLM-L6-v2')
    p.add_argument('--fit-queries', type=int, default=12)
    p.add_argument('--validation-queries', type=int, default=4)
    p.add_argument('--candidates-per-query', type=int, default=20)
    p.add_argument('--steps', type=int, default=120)
    p.add_argument('--allow-model-download', action='store_true',
                   help='Explicitly permit loading/downloading the pretrained encoder')
    args = p.parse_args()
    if min(args.fit_queries, args.validation_queries, args.candidates_per_query, args.steps) < 1:
        p.error('query/candidate limits and steps must be positive')
    groups, manifest = load_benchmark(args.train_data)
    if manifest.get('split') != 'train' or manifest.get('max_queries') is not None:
        p.error('full official SciFact training manifest required')
    fit_all, val_all = split_groups(tuple(groups), test_fraction=0.2, seed=42)
    if args.fit_queries > len(fit_all) or args.validation_queries > len(val_all):
        p.error('query limits exceed train-only split')
    fit = subset(fit_all, args.fit_queries, args.candidates_per_query)
    val = subset(val_all, args.validation_queries, args.candidates_per_query)
    if not args.allow_model_download and not args.embedding_cache.exists():
        p.error('embedding cache absent: pass --allow-model-download to enable pretrained encoder loading')

    def no_teacher_inference(_):
        raise RuntimeError('teacher cache incomplete; this study forbids teacher inference')

    teacher = score_groups(fit, scorer=no_teacher_inference, model_id=DEFAULT_MODEL,
                           cache_path=args.teacher_cache, batch_size=8)
    texts = list(dict.fromkeys(t for g in fit + val for t in
                               [g.query, *(c.passage for c in g.candidates)]))
    # Cache-only runs never instantiate the encoder. If entries are missing, fail explicitly.
    def encode_missing(batch):
        if not args.allow_model_download:
            raise RuntimeError('embedding cache incomplete; use --allow-model-download explicitly')
        nonlocal encoder
        if encoder is None:
            encoder = frozen_sentence_encoder(args.encoder)
        return encoder(batch)

    encoder = None
    start = time.perf_counter()
    embeddings = encode_cached(texts, encoder=encode_missing, model_id=args.encoder,
                               cache_path=args.embedding_cache)
    encoding_seconds = time.perf_counter() - start
    fit_x, val_x = text_feature_groups(fit, embeddings), text_feature_groups(val, embeddings)
    start = time.perf_counter()
    head = fit_text_head(fit, fit_x, teacher, steps=args.steps)
    training_seconds = time.perf_counter() - start
    scores = predict_text_head(val, val_x, head)
    report = {'experiment': '8A frozen text encoder feasibility', 'exploratory': True,
              'official_test_used': False, 'encoder': args.encoder, 'fit_queries': len(fit),
              'validation_queries': len(val), 'candidates_per_query_limit': args.candidates_per_query,
              'fit_candidates': sum(len(g.candidates) for g in fit),
              'fit_queries_with_retrieved_positive': sum(any(c.relevance is not None and c.relevance > 0 for c in g.candidates) for g in fit),
              'validation_queries_with_retrieved_positive': sum(any(c.relevance is not None and c.relevance > 0 for c in g.candidates) for g in val),
              'validation_candidates': sum(len(g.candidates) for g in val),
              'embedding_seconds': encoding_seconds, 'head_training_seconds': training_seconds,
              'head_parameters': head['parameter_count'],
              'text_student': summarize_ranking(val, scores, k=10),
              'bm25_same_subset': summarize_ranking(val, bm25_order_scores(val), k=10),
              'warning': 'Small truncated train-only subset; NOT comparable to 162-query top-100 validation results. Reused split; exploratory.'}
    for path, data in ((args.output, report), (args.model_output, {**head, 'encoder': args.encoder})):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'report': str(args.output), 'model': str(args.model_output), **report}, indent=2))


if __name__ == '__main__':
    main()
