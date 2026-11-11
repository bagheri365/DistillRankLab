"""Run paired, multi-seed ranking ablations and export raw results as CSV."""

import argparse

from distillranklab.ablation import AblationConfig, run_ablation, summarize, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='ablation_results.csv',
                        help='Path to output CSV (default: ablation_results.csv)')
    parser.add_argument('--seeds', type=int, default=5,
                        help='Number of independent seeds (default: 5)')
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error('--seeds must be positive')
    config = AblationConfig(seeds=tuple(range(11, 11 + args.seeds)))
    rows = run_ablation(config)
    write_csv(args.output, rows)
    print(f'Wrote {len(rows)} individual runs to {args.output}')
    print('Error  Temp  Method    Accuracy (mean±SD)   Δ vs hard (mean±SD)')
    for row in summarize(rows):
        print(f"{row['error_rate']:>4.0%}  {row['temperature']:>4.1f}  "
              f"{row['method']:<6}    "
              f"{row['accuracy_mean']:.3f} ± {row['accuracy_sd']:.3f}       "
              f"{row['delta_vs_hard_mean']:+.3f} ± {row['delta_vs_hard_sd']:.3f}")
    print('SD is across seeds; Δ is paired against the hard baseline for each seed.')
    print('Synthetic data only; small differences are not evidence of superiority.')


if __name__ == '__main__':
    main()
