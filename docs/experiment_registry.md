# Experiment registry

All scores below are **all-query** metrics on fixed BM25 top-100 candidate
pools, with unjudged passages treated as nonrelevant. The test split has
already been inspected repeatedly; all test findings are **exploratory**.

| Milestone | Split | Model / objective | nDCG@10 | MRR@10 | Notes |
|---|---|---|---:|---:|---|
| 6F | test (300) | BM25 | 0.663867 | 0.627618 | Fixed candidates |
| 6F | test (300) | Frozen cross-encoder | 0.680903 | 0.650915 | Difference vs BM25 not statistically significant (6G) |
| 7A | test (300) | Baseline linear, soft | 0.646147 | 0.619067 | 809 train queries |
| 7A | test (300) | Baseline linear, hybrid | 0.646166 | 0.619119 | 809 train queries |
| 7B | validation (162) | BM25 | 0.688365 | 0.652258 | Train-only validation |
| 7B | validation (162) | Selected baseline soft (T=2) | 0.682475 | 0.649081 | 7-trial selection; before refit |
| 7B | test (300) | Refit selected baseline soft | 0.644325 | 0.617758 | Exploratory final evaluation |
| 7C | validation (162) | Rich 8-feature hybrid | 0.682897 | 0.649647 | Selected among 9 trials; BM25 0.688365 |
| 7D | validation (162) | Linear vs 8/16-hidden MLP | TBD | TBD | Run `examples/16_scifact_nonlinear_ablation.py` |

## 7C run log (measured)

- Commit: `d7f1ff1`; environment: Intel Mac, Python 3.10.15; date and precise runtime not recorded
- Dataset manifest: official SciFact train, BM25 top-100, 809 queries
- Teacher: `cross-encoder/ms-marco-MiniLM-L-6-v2` (cached train-only scores)
- Split: 647 fit / 162 validation, SHA-256 seed 42
- Grid: baseline/lexical/rich × hard/soft/hybrid; 9 trials
- Optimization: learning rate 0.05, steps 120, temperature 2, hybrid hard weight 0.5
- Selected: rich 8-feature hybrid, hard weight 0.5; validation nDCG@10 **0.6828966118**, MRR@10 **0.6496472663**
- Wall-clock time: not recorded; hardware: Intel Mac CPU
- Observations: +0.000422 nDCG@10 vs best 4-feature soft model; still -0.005468 vs BM25. Small exploratory gain, not a significance claim.

**Rules:** Do not replace TBD with guessed results. Record negative findings.
Never use test-set scores to choose feature sets or optimization settings.

## 7D run log (fill in after execution)

- Commit SHA, date, hardware, Python/NumPy: **TBD**
- Dataset: official SciFact train, BM25 top-100; 647 fit / 162 validation
- Fixed controls: rich features, hybrid objective, T=2, hard weight=0.5, LR=0.05, steps=120, seed=42
- Linear (8 parameters): validation nDCG/MRR and runtime **TBD**
- MLP (8 hidden, 81 parameters): validation nDCG/MRR and runtime **TBD**
- MLP (16 hidden, 161 parameters): validation nDCG/MRR and runtime **TBD**
- Selected model and conclusion: **TBD**
- No official test data used; reused validation is exploratory.
