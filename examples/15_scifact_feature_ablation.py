"""Train-only feature-set ablation with fixed, query-disjoint SciFact validation."""
import argparse
import json
from pathlib import Path

from distillranklab.benchmark import load_benchmark
from distillranklab.real_student import FEATURE_SETS, save_student
from distillranklab.teacher import DEFAULT_MODEL, score_groups, cross_encoder_scorer
from distillranklab.validation import select_student


def ablation_configs():
    """Predeclared nine-trial grid; fixed optimization across feature sets."""
    return [dict(feature_set=feature_set, mode=mode, learning_rate=0.05,
                 steps=120, temperature=2.0, hard_weight=weight)
            for feature_set in FEATURE_SETS
            for mode, weight in (('hard', 1.0), ('soft', 0.0), ('hybrid', 0.5))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-data', type=Path, default=Path('datasets/scifact_full_train_bm25'))
    parser.add_argument('--train-cache', type=Path, default=Path('datasets/scifact_full_train_teacher_scores.json'))
    parser.add_argument('--output', type=Path, default=Path('datasets/scifact_feature_ablation.json'))
    parser.add_argument('--student-output', type=Path, default=Path('datasets/scifact_feature_student.json'))
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--training-seed', type=int, default=42)
    parser.add_argument('--allow-inference', action='store_true')
    args = parser.parse_args()
    groups, manifest = load_benchmark(args.train_data)
    if manifest.get('split') != 'train':
        parser.error('official SciFact train split required')
    scorer = None

    def score_missing(pairs):
        nonlocal scorer
        if not args.allow_inference:
            raise RuntimeError('Teacher cache incomplete; --allow-inference required to score missing pairs')
        if scorer is None:
            scorer = cross_encoder_scorer(DEFAULT_MODEL, device='cpu')
        return scorer(pairs)

    teacher_scores = score_groups(groups, scorer=score_missing, model_id=DEFAULT_MODEL,
                                  cache_path=args.train_cache, batch_size=8)
    report, model = select_student(groups, teacher_scores, configs=ablation_configs(),
                                   split_seed=args.split_seed, training_seed=args.training_seed)
    report.update({'experiment': '7C feature-set ablation', 'dataset': 'BEIR SciFact',
                   'manifest': manifest, 'feature_sets': {k: list(v) for k, v in FEATURE_SETS.items()},
                   'teacher_model': DEFAULT_MODEL, 'test_data_used': False,
                   'predeclared_grid': '3 feature sets x hard/soft/hybrid, fixed learning rate and steps',
                   'caution': 'Exploratory validation; no untouched test claims; candidate order is a feature'})
    model['selection'] = {'experiment': '7C', 'config': report['selected_config'],
                          'split_seed': args.split_seed}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    save_student(model, args.student_output)
    print(json.dumps({'report': str(args.output), 'student': str(args.student_output),
                      'fit_queries': report['fit_queries'], 'validation_queries': report['validation_queries'],
                      'trials': report['trials'], 'selected_config': report['selected_config'],
                      'selected_validation_ndcg_at_k_all_queries': report['selected_validation_ndcg_at_k_all_queries'],
                      'validation_bm25_ndcg_at_k_all_queries': report['validation_bm25']['ndcg_at_k_all_queries']}, indent=2))


if __name__ == '__main__':
    main()
