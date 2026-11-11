"""Evaluate frozen teacher rankings on the illustrative query/passage fixture."""

import argparse
from pathlib import Path

from distillranklab.ranking_metrics import evaluate_ranking
from distillranklab.real_data import load_groups, split_groups
from distillranklab.teacher import DEFAULT_MODEL, cross_encoder_scorer, score_groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real', action='store_true', help='run pretrained CPU cross-encoder')
    parser.add_argument('--cache', type=Path, default=None)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--unjudged', choices=('nonrelevant', 'judged_only'),
                        default='nonrelevant')
    args = parser.parse_args()
    base = Path(__file__).resolve().parents[1] / 'tests' / 'fixtures' / 'tiny_ranking'
    groups = load_groups(*(base / name for name in
                           ('queries.tsv', 'passages.tsv', 'candidates.tsv', 'qrels.tsv')))
    _, test = split_groups(groups, test_fraction=0.4, seed=42)
    if args.real:
        model_id = DEFAULT_MODEL
        scorer = cross_encoder_scorer(model_id, device='cpu')
    else:
        model_id = 'demo-token-overlap-v1'

        def scorer(pairs):
            return [float(len(set(q.lower().split()) & set(p.lower().split())))
                    for q, p in pairs]

    cache = args.cache or Path('teacher_scores_real.json' if args.real
                               else 'teacher_scores_demo.json')
    scores = score_groups(test, scorer=scorer, model_id=model_id,
                          cache_path=cache, batch_size=args.batch_size)
    result = evaluate_ranking(test, scores, k=10, unjudged=args.unjudged)
    print('ILLUSTRATIVE FIXTURE — not MS MARCO; metrics are not benchmark results')
    print(f'Model: {model_id}; held-out queries: {len(test)}; policy: {args.unjudged}')
    print(f'MRR@10: {result["mrr"]:.4f}; nDCG@10: {result["ndcg"]:.4f}')
    print(f'Eligible queries: {result["eligible_queries"]}; '
          f'excluded without judged positive: {result["excluded_no_positive"]}')
    print(f'Judged candidates: {result["judged_candidates"]}; '
          f'unjudged: {result["unjudged_candidates"]}; cache: {cache}')


if __name__ == '__main__':
    main()
