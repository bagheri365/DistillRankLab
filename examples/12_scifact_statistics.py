"""Compute paired confidence intervals and randomization p-values without inference."""
import argparse
import json
from pathlib import Path

from distillranklab.statistics import paired_statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--per-query', type=Path, required=True,
                        help='JSON produced by examples/10_scifact_reranking.py --per-query-output')
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--replicates', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--confidence', type=float, default=0.95)
    args = parser.parse_args()
    diagnostics = json.loads(args.per_query.read_text(encoding='utf-8'))
    report = paired_statistics(diagnostics, replicates=args.replicates,
                               seed=args.seed, confidence=args.confidence)
    formatted = json.dumps(report, indent=2) + '\n'
    print(formatted, end='')
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(formatted, encoding='utf-8')


if __name__ == '__main__':
    main()
