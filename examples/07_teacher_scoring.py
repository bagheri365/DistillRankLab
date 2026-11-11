"""Demonstrate cached teacher scoring; --real downloads an optional model."""

import argparse
from pathlib import Path

from distillranklab.real_data import load_groups
from distillranklab.teacher import DEFAULT_MODEL, cross_encoder_scorer, score_groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real', action='store_true', help='use pretrained cross-encoder on CPU')
    parser.add_argument('--cache', type=Path, default=Path('teacher_scores_demo.json'))
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    fixture = Path(__file__).resolve().parents[1] / 'tests' / 'fixtures' / 'tiny_ranking'
    groups = load_groups(*(fixture / name for name in
                           ('queries.tsv', 'passages.tsv', 'candidates.tsv', 'qrels.tsv')))
    if args.real:
        model_id = DEFAULT_MODEL
        scorer = cross_encoder_scorer(model_id, device='cpu')
    else:
        model_id = 'demo-token-overlap-v1'

        def scorer(pairs):
            return [float(len(set(q.lower().split()) & set(p.lower().split())))
                    for q, p in pairs]

    scores = score_groups(groups, scorer=scorer, model_id=model_id,
                          cache_path=args.cache, batch_size=args.batch_size)
    print(f'Model: {model_id}; scored candidates: {len(scores)}; cache: {args.cache}')
    for group in groups:
        best = max(group.candidates, key=lambda c: scores[(group.query_id, c.passage_id)])
        print(f'{group.query_id}: top passage {best.passage_id} '
              f'(score={scores[(group.query_id, best.passage_id)]:.3f})')
    if not args.real:
        print('Demo scorer only; use --real for actual pretrained teacher scores.')


if __name__ == '__main__':
    main()
