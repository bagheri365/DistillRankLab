"""Train a CPU-only listwise student on official SciFact train queries only."""
import argparse
import json
from pathlib import Path

from distillranklab.benchmark import load_benchmark, bm25_order_scores, summarize_ranking
from distillranklab.real_student import fit_student, save_student, evaluate_student
from distillranklab.teacher import DEFAULT_MODEL, cross_encoder_scorer, score_groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-data', type=Path, default=Path('datasets/scifact_train_bm25'))
    parser.add_argument('--test-data', type=Path, default=None,
                        help='Optional held-out evaluation; NEVER used in fitting')
    parser.add_argument('--train-cache', type=Path, default=Path('datasets/scifact_train_teacher_scores.json'))
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--mode', choices=('soft', 'hard', 'hybrid'), default='soft')
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--hard-weight', type=float, default=0.5)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--output', type=Path, default=Path('datasets/scifact_student.json'))
    args = parser.parse_args()
    train, manifest = load_benchmark(args.train_data)
    if manifest['split'] != 'train':
        parser.error('--train-data must use the official SciFact train split')
    test = None
    if args.test_data is not None:
        test, test_manifest = load_benchmark(args.test_data)
        if test_manifest['split'] != 'test':
            parser.error('--test-data must use the official SciFact test split')
        if set(g.query_id for g in train) & set(g.query_id for g in test):
            parser.error('train/test query ID overlap')
    need_teacher = args.mode != 'hard' and not (args.mode == 'hybrid' and args.hard_weight == 1)
    scores = None
    if need_teacher:
        scorer = cross_encoder_scorer(args.model, device='cpu')
        scores = score_groups(train, scorer=scorer, model_id=args.model,
                              cache_path=args.train_cache, batch_size=args.batch_size)
    model = fit_student(train, scores, mode=args.mode, temperature=args.temperature,
                        hard_weight=args.hard_weight, steps=args.steps,
                        learning_rate=args.learning_rate, seed=args.seed)
    model['training_dataset'] = 'BEIR SciFact official train'
    model['teacher_model'] = args.model if need_teacher else None
    model['train_retriever'] = manifest['retriever']
    save_student(model, args.output)
    report = {'student_model': str(args.output), 'mode': args.mode,
              'train_queries': len(train), 'train_metrics': evaluate_student(train, model)}
    if test is not None:
        report['test_queries'] = len(test)
        report['test_bm25'] = summarize_ranking(test, bm25_order_scores(test))
        report['test_student'] = evaluate_student(test, model)
        report['test_teacher'] = 'Not evaluated here; use existing frozen-teacher benchmark results'
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
