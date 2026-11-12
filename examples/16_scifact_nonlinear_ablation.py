"""Train-only 7D linear-vs-MLP capacity comparison; never loads SciFact test."""
import argparse
import json
from pathlib import Path
from time import perf_counter

from distillranklab.benchmark import bm25_order_scores, load_benchmark, summarize_ranking
from distillranklab.nonlinear_student import fit_nonlinear, predict_nonlinear
from distillranklab.real_data import split_groups
from distillranklab.real_student import fit_student, predict_student, save_student
from distillranklab.teacher import DEFAULT_MODEL, cross_encoder_scorer, score_groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-data', type=Path, default=Path('datasets/scifact_full_train_bm25'))
    p.add_argument('--train-cache', type=Path, default=Path('datasets/scifact_full_train_teacher_scores.json'))
    p.add_argument('--output', type=Path, default=Path('datasets/scifact_nonlinear_ablation.json'))
    p.add_argument('--student-output', type=Path, default=Path('datasets/scifact_nonlinear_student.json'))
    p.add_argument('--split-seed', type=int, default=42)
    p.add_argument('--training-seed', type=int, default=42)
    p.add_argument('--steps', type=int, default=120)
    p.add_argument('--allow-inference', action='store_true')
    args = p.parse_args()
    groups, manifest = load_benchmark(args.train_data)
    if manifest.get('dataset') != 'BEIR SciFact' or manifest.get('split') != 'train':
        p.error('official SciFact train split required')
    scorer = None

    def score_missing(pairs):
        nonlocal scorer
        if not args.allow_inference:
            raise RuntimeError('Incomplete teacher cache; pass --allow-inference explicitly')
        if scorer is None:
            scorer = cross_encoder_scorer(DEFAULT_MODEL, device='cpu')
        return scorer(pairs)

    teacher = score_groups(groups, scorer=score_missing, model_id=DEFAULT_MODEL,
                           cache_path=args.train_cache, batch_size=8)
    fit, validation = split_groups(tuple(groups), test_fraction=0.2, seed=args.split_seed)
    configs = [
        {'architecture': 'linear', 'feature_set': 'rich', 'mode': 'hybrid', 'hard_weight': 0.5},
        {'architecture': 'mlp', 'feature_set': 'rich', 'mode': 'hybrid', 'hard_weight': 0.5, 'hidden_units': 8},
        {'architecture': 'mlp', 'feature_set': 'rich', 'mode': 'hybrid', 'hard_weight': 0.5, 'hidden_units': 16},
    ]
    trials = []
    for index, config in enumerate(configs):
        t0 = perf_counter()
        common = dict(mode=config['mode'], feature_set=config['feature_set'], hard_weight=0.5,
                      temperature=2.0, learning_rate=0.05, steps=args.steps, seed=args.training_seed)
        if config['architecture'] == 'linear':
            model = fit_student(fit, teacher, **common)
            predictions = predict_student(validation, model)
            parameters = len(model['weights'])
        else:
            model = fit_nonlinear(fit, teacher, hidden_units=config['hidden_units'], **common)
            predictions = predict_nonlinear(validation, model)
            parameters = model['parameter_count']
        metrics = summarize_ranking(validation, predictions, k=10)
        trials.append({'index': index, 'config': config, 'parameter_count': parameters,
                       'wall_seconds_fit_and_validation': perf_counter() - t0,
                       'validation_mrr_at_10': metrics['mrr_at_k_all_queries'],
                       'validation_ndcg_at_10': metrics['ndcg_at_k_all_queries']})
    best = max(range(len(trials)), key=lambda i: (trials[i]['validation_ndcg_at_10'], -i))
    chosen = configs[best]
    common = dict(mode='hybrid', feature_set='rich', hard_weight=0.5, temperature=2.0,
                  learning_rate=0.05, steps=args.steps, seed=args.training_seed)
    if chosen['architecture'] == 'linear':
        model = fit_student(groups, teacher, **common)
    else:
        model = fit_nonlinear(groups, teacher, hidden_units=chosen['hidden_units'], **common)
    model['selection'] = {'experiment': '7D', 'split_seed': args.split_seed,
                          'selected_index': best, 'config': chosen}
    report = {'experiment': '7D nonlinear capacity ablation', 'manifest': manifest,
              'fit_queries': len(fit), 'validation_queries': len(validation),
              'fit_query_ids': sorted(g.query_id for g in fit),
              'validation_query_ids': sorted(g.query_id for g in validation),
              'bm25_validation_ndcg_at_10': summarize_ranking(validation, bm25_order_scores(validation))['ndcg_at_k_all_queries'],
              'trials': trials, 'selected_index': best, 'selected_config': chosen,
              'teacher_model': DEFAULT_MODEL, 'test_used': False,
              'caution': 'Exploratory reuse of validation split; no test-based tuning; unjudged treated nonrelevant'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    save_student(model, args.student_output)
    print(json.dumps({'report': str(args.output), 'student': str(args.student_output),
                      'fit_queries': len(fit), 'validation_queries': len(validation),
                      'bm25_validation_ndcg_at_10': report['bm25_validation_ndcg_at_10'],
                      'trials': trials, 'selected_config': chosen}, indent=2))


if __name__ == '__main__':
    main()
