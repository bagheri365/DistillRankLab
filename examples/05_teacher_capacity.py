"""Teacher corruption x student feature-capacity ablation (NumPy only)."""

import argparse

from distillranklab.capacity import (CapacityConfig, run_capacity_ablation,
                                     summarize_capacity, write_capacity_csv)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=5,
                        help='number of seeds (1 to 5; default 5)')
    parser.add_argument('--output', default='teacher_capacity_results.csv')
    args = parser.parse_args()
    if not 1 <= args.seeds <= 5:
        parser.error('--seeds must be between 1 and 5')
    config = CapacityConfig(seeds=(11, 12, 13, 14, 15)[:args.seeds])
    rows = run_capacity_ablation(config)
    write_capacity_csv(args.output, rows)
    print(f'Wrote {len(rows)} seed-level rows to {args.output}')
    print('Cap  Error  Temp  Method    Accuracy (mean±SD)    Δ vs hard (mean±SD)')
    for r in summarize_capacity(rows):
        print(f"{r['capacity']:>3} {r['error_rate']:>5.0%} {r['temperature']:>5.1f}  "
              f"{r['method']:<7}  {r['accuracy_mean']:.3f} ± {r['accuracy_sd']:.3f}"
              f"        {r['delta_vs_hard_mean']:+.3f} ± {r['delta_vs_hard_sd']:.3f}")
    print('Capacity means number of visible input features, not model width.')
    print('Synthetic results; small differences do not establish superiority.')


if __name__ == '__main__':
    main()
