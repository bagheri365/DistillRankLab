"""8D: replay 8C models and audit candidate alignment and judgment sensitivity."""
import argparse
import json
from pathlib import Path

from distillranklab.benchmark import bm25_order_scores, load_benchmark
from distillranklab.evaluation_audit import audit_pair, candidate_fingerprint
from distillranklab.nonlinear_student import fit_nonlinear, predict_nonlinear
from distillranklab.real_data import split_groups
from distillranklab.scaled_text import cached_embeddings_sqlite
from distillranklab.semantic_ablation import run_feature_ablation
from distillranklab.teacher import DEFAULT_MODEL, score_groups
from distillranklab.text_student import text_feature_groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-data', type=Path, default=Path('datasets/scifact_full_train_bm25'))
    p.add_argument('--teacher-cache', type=Path, default=Path('datasets/scifact_full_train_teacher_scores.json'))
    p.add_argument('--embedding-cache', type=Path, default=Path('datasets/scifact_8b_embeddings.sqlite'))
    p.add_argument('--output', type=Path, default=Path('datasets/scifact_8d_audit.json'))
    p.add_argument('--encoder', default='sentence-transformers/all-MiniLM-L6-v2')
    args = p.parse_args()
    groups, manifest = load_benchmark(args.train_data)
    if (manifest.get('split') != 'train' or manifest.get('max_queries') is not None
            or len(groups) != 809 or any(len(g.candidates) != 100 for g in groups)):
        p.error('full official SciFact TRAIN top-100 required; no test access')
    fit, val = split_groups(tuple(groups), test_fraction=0.2, seed=42)
    if len(fit) != 647 or len(val) != 162 or set(g.query_id for g in fit) & set(g.query_id for g in val):
        p.error('unexpected or overlapping query split')

    def forbidden_teacher(_):
        raise RuntimeError('teacher cache incomplete: inference prohibited')

    teacher = score_groups(fit, scorer=forbidden_teacher, model_id=DEFAULT_MODEL,
                           cache_path=args.teacher_cache, batch_size=16)
    texts = list(dict.fromkeys(t for g in fit + val for t in [g.query, *(c.passage for c in g.candidates)]))

    def forbidden_encoder(_):
        raise RuntimeError('embedding cache incomplete: inference prohibited')

    embeddings, cache = cached_embeddings_sqlite(texts, model_id=args.encoder,
                                                  path=args.embedding_cache, encoder=forbidden_encoder, log=None)
    if cache['new_embeddings']:
        raise RuntimeError('unexpected new embeddings')
    fit_x, val_x = text_feature_groups(fit, embeddings), text_feature_groups(val, embeddings)
    feature_blocks, predictions = run_feature_ablation(fit, val, fit_x, val_x, teacher)
    mlp = fit_nonlinear(fit, teacher, hidden_units=8, mode='hybrid', feature_set='rich',
                        temperature=2.0, hard_weight=0.5, steps=120, learning_rate=0.05, seed=42)
    scores = {'bm25': bm25_order_scores(val), 'mlp8': predict_nonlinear(val, mlp), **predictions}
    comparisons = [('combined', 'bm25'), ('combined', 'mlp8'),
                   ('semantic_only', 'lexical_only'), ('combined', 'semantic_only'),
                   ('combined', 'lexical_only'), ('semantic_only', 'bm25')]
    audits = {f'{challenger}_vs_{baseline}': audit_pair(
        val, scores[baseline], scores[challenger],
        baseline_name=baseline, challenger_name=challenger)
        for challenger, baseline in comparisons}
    report = {'experiment': '8D fixed-pool evaluation audit', 'exploratory': True,
              'official_test_used': False, 'fit_queries': len(fit), 'validation_queries': len(val),
              'fit_pool_sha256': candidate_fingerprint(fit),
              'validation_pool_sha256': candidate_fingerprint(val),
              'embedding_cache': cache, 'feature_blocks_nonrelevant': feature_blocks,
              'comparisons': audits,
              'warning': ('Repeatedly reused validation; no independent confirmation. '
                          'Judged-only discards unjudged candidates and is not an unbiased alternative. '
                          'No multiple-testing correction; no official test access.')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    compact = {'report': str(args.output), 'experiment': report['experiment'],
               'official_test_used': False, 'embedding_cache': cache,
               'validation_pool_sha256': report['validation_pool_sha256'], 'comparisons': {}}
    for name, audit in audits.items():
        compact['comparisons'][name] = {policy: {
            'delta_ndcg': outcome['delta_distribution']['ndcg']['mean_delta'],
            'delta_mrr': outcome['delta_distribution']['mrr']['mean_delta'],
            'ndcg_improved': outcome['delta_distribution']['ndcg']['improved_queries'],
            'ndcg_regressed': outcome['delta_distribution']['ndcg']['regressed_queries'],
            'ndcg_sd': outcome['delta_distribution']['ndcg']['sample_sd'],
            'aggregate_verified': outcome['per_query_aggregate_verified']}
            for policy, outcome in audit['policies'].items()}
    print(json.dumps(compact, indent=2))


if __name__ == '__main__':
    main()
