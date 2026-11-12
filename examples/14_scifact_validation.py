"""Validation-only hyperparameter selection for SciFact linear students."""
import argparse
import json
from pathlib import Path

from distillranklab.benchmark import load_benchmark, bm25_order_scores, summarize_ranking
from distillranklab.real_student import evaluate_student, save_student
from distillranklab.teacher import DEFAULT_MODEL, cross_encoder_scorer, score_groups
from distillranklab.validation import make_configs, select_student


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-data', type=Path, default=Path('datasets/scifact_full_train_bm25'))
    parser.add_argument('--train-cache', type=Path, default=Path('datasets/scifact_full_train_teacher_scores.json'))
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--learning-rates', type=float, nargs='+', default=[0.05])
    parser.add_argument('--temperatures', type=float, nargs='+', default=[1.0, 2.0])
    parser.add_argument('--hard-weights', type=float, nargs='+', default=[0.25, 0.5])
    parser.add_argument('--steps', type=int, nargs='+', default=[120])
    parser.add_argument('--validation-fraction', type=float, default=0.2)
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--training-seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--allow-inference', action='store_true',
                        help='Permit CPU teacher inference on missing cache entries; otherwise fail')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--output', type=Path, default=Path('datasets/scifact_validation_report.json'))
    parser.add_argument('--student-output', type=Path, default=Path('datasets/scifact_selected_student.json'))
    parser.add_argument('--test-data', type=Path, default=None,
                        help='OPTIONAL final evaluation only; NEVER used for model selection')
    args = parser.parse_args()

    groups, manifest = load_benchmark(args.train_data)
    if manifest['split'] != 'train':
        parser.error('selection requires the official train split')
    configs = make_configs(learning_rates=args.learning_rates, temperatures=args.temperatures,
                           hard_weights=args.hard_weights, steps=args.steps)
    need_teacher = any(c['mode'] != 'hard' for c in configs)
    teacher_scores = None
    if need_teacher:
        lazy_scorer = None

        def score_missing(pairs):
            nonlocal lazy_scorer
            if not args.allow_inference:
                raise RuntimeError('Teacher cache is incomplete; rerun with --allow-inference to score missing pairs')
            if lazy_scorer is None:
                lazy_scorer = cross_encoder_scorer(args.model, device='cpu')
            return lazy_scorer(pairs)

        teacher_scores = score_groups(groups, scorer=score_missing, model_id=args.model,
                                      cache_path=args.train_cache, batch_size=args.batch_size)
    report, student = select_student(groups, teacher_scores, configs=configs,
                                      validation_fraction=args.validation_fraction,
                                      split_seed=args.split_seed, training_seed=args.training_seed,
                                      k=args.k)
    report.update({'dataset': 'BEIR SciFact', 'official_split': 'train',
                   'train_manifest': manifest, 'teacher_model': args.model if need_teacher else None,
                   'unjudged_policy': 'nonrelevant', 'candidate_pool': 'fixed BM25',
                   'test_used_for_selection': False})
    if args.test_data is not None:
        test, test_manifest = load_benchmark(args.test_data)
        if test_manifest['split'] != 'test':
            parser.error('--test-data must be the official test split')
        if {g.query_id for g in groups} & {g.query_id for g in test}:
            parser.error('train/test query overlap')
        report['final_test'] = {
            'query_count': len(test),
            'bm25': summarize_ranking(test, bm25_order_scores(test), k=args.k),
            'selected_student': evaluate_student(test, student, k=args.k),
            'warning': 'Final evaluation only; do not tune or select models from these scores',
        }
    student['selection'] = {'validation_metric': report['selection_metric'],
                            'selected_config': report['selected_config'],
                            'split_seed': args.split_seed,
                            'validation_fraction': args.validation_fraction}
    save_student(student, args.student_output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'report': str(args.output), 'student': str(args.student_output),
                      'fit_queries': report['fit_queries'],
                      'validation_queries': report['validation_queries'],
                      'trials': len(report['trials']),
                      'selected_config': report['selected_config'],
                      'selected_validation_ndcg_at_k_all_queries': report['selected_validation_ndcg_at_k_all_queries'],
                      'validation_bm25_ndcg_at_k_all_queries': report['validation_bm25']['ndcg_at_k_all_queries'],
                      'final_test': report.get('final_test')}, indent=2))


if __name__ == '__main__':
    main()
