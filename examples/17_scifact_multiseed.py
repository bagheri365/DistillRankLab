"""Train-only 7E multi-seed robustness experiment; no official test data."""
import argparse
import json
from pathlib import Path

from distillranklab.benchmark import load_benchmark
from distillranklab.real_data import split_groups
from distillranklab.robustness import run_robustness
from distillranklab.teacher import DEFAULT_MODEL, cross_encoder_scorer, score_groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-data', type=Path, default=Path('datasets/scifact_full_train_bm25'))
    p.add_argument('--train-cache', type=Path, default=Path('datasets/scifact_full_train_teacher_scores.json'))
    p.add_argument('--output', type=Path, default=Path('datasets/scifact_multiseed.json'))
    p.add_argument('--split-seed', type=int, default=42)
    p.add_argument('--seeds', type=int, nargs='+', default=[11, 23, 42, 67, 101])
    p.add_argument('--steps', type=int, default=120)
    p.add_argument('--replicates', type=int, default=10000)
    p.add_argument('--bootstrap-seed', type=int, default=42)
    p.add_argument('--allow-inference', action='store_true')
    args = p.parse_args()
    groups, manifest = load_benchmark(args.train_data)
    if manifest.get('split') != 'train' or manifest.get('max_queries') is not None:
        p.error('full official SciFact train split required')
    scorer = None

    def score_missing(pairs):
        nonlocal scorer
        if not args.allow_inference:
            raise RuntimeError('Teacher cache incomplete; pass --allow-inference explicitly')
        if scorer is None:
            scorer = cross_encoder_scorer(DEFAULT_MODEL, device='cpu')
        return scorer(pairs)

    teacher = score_groups(groups, scorer=score_missing, model_id=DEFAULT_MODEL,
                           cache_path=args.train_cache, batch_size=8)
    fit, validation = split_groups(tuple(groups), test_fraction=0.2, seed=args.split_seed)
    report = run_robustness(fit, validation, teacher, seeds=args.seeds,
                            steps=args.steps, replicates=args.replicates,
                            bootstrap_seed=args.bootstrap_seed)
    report.update({'experiment': '7E multi-seed robustness', 'manifest': manifest,
                   'split_seed': args.split_seed, 'teacher_model': DEFAULT_MODEL,
                   'fit_query_ids': sorted(g.query_id for g in fit)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'report': str(args.output), 'seeds': report['seeds'],
                      'fit_queries': report['fit_queries'],
                      'validation_queries': report['validation_queries'],
                      'bm25_validation_ndcg_at_10': report['bm25_validation_ndcg_at_10'],
                      'seed_summaries': report['seed_summaries'],
                      'paired_query_comparisons': report['paired_query_comparisons'],
                      'selected_architecture': None}, indent=2))


if __name__ == '__main__':
    main()
