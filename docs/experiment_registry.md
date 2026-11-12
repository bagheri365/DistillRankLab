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
| 7D | validation (162) | Linear, rich hybrid | 0.682897 | 0.649647 | 8 parameters; 45.22 s |
| 7D | validation (162) | MLP-8, rich hybrid | 0.684157 | 0.650958 | 81 parameters; 50.37 s |
| 7D | validation (162) | MLP-16, rich hybrid | 0.684518 | 0.651352 | 161 parameters; 53.14 s; selected |
| 7E | validation (162) | Linear, 5-seed mean | 0.682640 | 0.649339 | nDCG sample SD 0.000234; 45.38 s mean |
| 7E | validation (162) | MLP-8, 5-seed mean | 0.684437 | 0.651298 | nDCG sample SD 0.000175; 50.27 s mean |
| 7E | validation (162) | MLP-16, 5-seed mean | 0.684408 | 0.651232 | nDCG sample SD 0.000185; 53.60 s mean |

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

## 7D run log (measured)

- Commit: `ffd8277`; hardware: Intel Mac CPU, Python 3.10.15; NumPy version/date not recorded
- Dataset: official SciFact train, BM25 top-100; 647 fit / 162 validation
- Fixed controls: rich features, hybrid objective, T=2, hard weight=0.5, LR=0.05, steps=120, seed=42
- Linear (8 parameters): nDCG **0.6828966118**, MRR **0.6496472663**, 45.22 s
- MLP-8 (81 parameters): nDCG **0.6841569805**, MRR **0.6509577699**, 50.37 s
- MLP-16 (161 parameters): nDCG **0.6845179961**, MRR **0.6513521458**, 53.14 s
- BM25 validation nDCG **0.6883649753**; MLP-16 selected, +0.001621 vs linear
- No official test data used; reused validation is exploratory. Times include fit and validation, not teacher scoring.

## 7E run log (measured)

- Commit: `2fc8445`; Intel Mac CPU, Python 3.10.15; total runtime not recorded
- Fixed split: 647 fit / 162 validation; training seeds 11, 23, 42, 67, 101
- Mean nDCG@10 (sample SD): linear **0.6826398583** (0.0002343828); MLP-8 **0.6844370842** (0.0001748424); MLP-16 **0.6844079588** (0.0001854219)
- Mean MRR@10: linear **0.6493386243**; MLP-8 **0.6512977660**; MLP-16 **0.6512321184**
- Mean fit + validation wall time: linear **45.38 s**; MLP-8 **50.27 s**; MLP-16 **53.60 s**
- BM25 validation nDCG@10 **0.6883649753**; both MLP means below BM25
- Paired MLP-8 minus linear: nDCG delta **+0.0017972259**, 95% bootstrap CI **[-0.0061102113, 0.0107699772]**, sign-flip p **0.6980**; MRR delta **+0.0019591417**, CI **[-0.0052028219, 0.0094591417]**, p **0.6180**
- Paired MLP-16 minus linear: nDCG delta **+0.0017681005**, 95% bootstrap CI **[-0.0061303836, 0.0107289729]**, sign-flip p **0.7016**; MRR delta **+0.0018934940**, CI **[-0.0052226631, 0.0093449931]**, p **0.6341**
- Both nDCG intervals contain zero; no significant improvement established. MLP-8 slightly exceeds MLP-16 on mean nDCG; neither beats BM25. All comparisons exploratory and uncorrected for multiple comparisons.
- No model selection and no official test evaluation. Record null or negative findings.
