"""8C: train-only lexical/semantic/combined feature ablation with paired statistics."""
import argparse
import json
from pathlib import Path
import time

from distillranklab.benchmark import bm25_order_scores, load_benchmark, summarize_ranking
from distillranklab.nonlinear_student import fit_nonlinear, predict_nonlinear
from distillranklab.real_data import split_groups
from distillranklab.scaled_text import cached_embeddings_sqlite
from distillranklab.semantic_ablation import compare_paired, run_feature_ablation
from distillranklab.teacher import DEFAULT_MODEL, score_groups
from distillranklab.text_student import frozen_sentence_encoder, text_feature_groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-data', type=Path, default=Path('datasets/scifact_full_train_bm25'))
    p.add_argument('--teacher-cache', type=Path, default=Path('datasets/scifact_full_train_teacher_scores.json'))
    p.add_argument('--embedding-cache', type=Path, default=Path('datasets/scifact_8b_embeddings.sqlite'))
    p.add_argument('--output', type=Path, default=Path('datasets/scifact_8c_ablation.json'))
    p.add_argument('--encoder', default='sentence-transformers/all-MiniLM-L6-v2')
    p.add_argument('--allow-model-download', action='store_true')
    p.add_argument('--steps', type=int, default=120)
    p.add_argument('--replicates', type=int, default=10000)
    args = p.parse_args()
    if args.steps < 1 or args.replicates < 100:
        p.error('steps must be positive and replicates >= 100')
    groups, manifest = load_benchmark(args.train_data)
    if (manifest.get('split') != 'train' or manifest.get('max_queries') is not None
            or len(groups) != 809 or any(len(g.candidates) != 100 for g in groups)):
        p.error('full official SciFact train top-100 benchmark required')
    fit, val = split_groups(tuple(groups), test_fraction=0.2, seed=42)
    if len(fit) != 647 or len(val) != 162:
        p.error('unexpected query split')

    def forbidden_teacher(_):
        raise RuntimeError('teacher cache incomplete; inference prohibited')

    teacher = score_groups(fit, scorer=forbidden_teacher, model_id=DEFAULT_MODEL,
                           cache_path=args.teacher_cache, batch_size=16)
    texts = list(dict.fromkeys(t for g in fit + val for t in [g.query, *(c.passage for c in g.candidates)]))
    encoder = None

    def encode_missing(batch):
        nonlocal encoder
        if not args.allow_model_download:
            raise RuntimeError('embedding cache incomplete; use --allow-model-download')
        if encoder is None:
            encoder = frozen_sentence_encoder(args.encoder, batch_size=32)
        return encoder(batch)

    embeddings, cache_info = cached_embeddings_sqlite(texts, model_id=args.encoder,
                                                       path=args.embedding_cache, encoder=encode_missing)
    started = time.perf_counter()
    fit_x, val_x = text_feature_groups(fit, embeddings), text_feature_groups(val, embeddings)
    feature_seconds = time.perf_counter() - started
    started = time.perf_counter()
    ablation, predictions = run_feature_ablation(fit, val, fit_x, val_x, teacher, steps=args.steps)
    ablation_seconds = time.perf_counter() - started
    config = dict(mode='hybrid', feature_set='rich', temperature=2.0,
                  hard_weight=0.5, steps=args.steps, learning_rate=0.05, seed=42)
    mlp = fit_nonlinear(fit, teacher, hidden_units=8, **config)
    mlp_scores = predict_nonlinear(val, mlp)
    bm25_scores = bm25_order_scores(val)
    pairs = [('combined', 'bm25', bm25_scores, predictions['combined']),
             ('combined', 'mlp8', mlp_scores, predictions['combined']),
             ('semantic_only', 'lexical_only', predictions['lexical_only'], predictions['semantic_only']),
             ('combined', 'semantic_only', predictions['semantic_only'], predictions['combined']),
             ('combined', 'lexical_only', predictions['lexical_only'], predictions['combined'])]
    comparisons = {f'{challenger}_vs_{baseline}': compare_paired(
        val, base_scores, challenger_scores, baseline_name=baseline,
        challenger_name=challenger, replicates=args.replicates)
        for challenger, baseline, base_scores, challenger_scores in pairs}
    report = {'experiment': '8C semantic feature block ablation', 'exploratory': True,
              'official_test_used': False, 'fit_queries': len(fit), 'validation_queries': len(val),
              'candidate_pool': 'BM25 top-100', 'steps': args.steps, 'seed': 42,
              'encoder': args.encoder, 'encoder_frozen': True, 'embedding_cache': cache_info,
              'feature_construction_seconds': feature_seconds,
              'ablation_fit_and_validation_seconds': ablation_seconds,
              'bm25': summarize_ranking(val, bm25_scores, k=10),
              'mlp8': summarize_ranking(val, mlp_scores, k=10),
              'feature_blocks': ablation, 'paired_comparisons': comparisons,
              'warning': 'Reused train-only validation; multiple exploratory comparisons; no official test data.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'report': str(args.output), **report}, indent=2))


if __name__ == '__main__':
    main()
