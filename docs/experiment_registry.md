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
| 7C | validation (162) | Feature ablation | TBD | TBD | Run `examples/15_scifact_feature_ablation.py`; fill in observed scores |

## 7C run log (fill in after execution)

- Date, commit SHA, environment: **TBD**
- Dataset manifest: official SciFact train, BM25 top-100, 809 queries
- Teacher: `cross-encoder/ms-marco-MiniLM-L-6-v2` (cached train-only scores)
- Split: 647 fit / 162 validation, SHA-256 seed 42
- Grid: baseline/lexical/rich × hard/soft/hybrid; 9 trials
- Optimization: learning rate 0.05, steps 120, temperature 2, hybrid hard weight 0.5
- Selected configuration, validation nDCG@10, MRR@10: **TBD**
- Wall-clock time and hardware: **TBD**
- Observations, failures, interpretation: **TBD**

**Rules:** Do not replace TBD with guessed results. Record negative findings.
Never use test-set scores to choose feature sets or optimization settings.
