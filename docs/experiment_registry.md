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

## 8A frozen semantic encoder feasibility (original preregistered plan)

- Status at planning: **TBD**; actual results are recorded below in the measured 8A run log.
- Frozen encoder: `sentence-transformers/all-MiniLM-L6-v2`; 10-feature linear
  ranking head (8 rich lexical + 2 pretrained embedding interaction features).
- Teacher: cached `cross-encoder/ms-marco-MiniLM-L-6-v2` training scores only.
- Scope: first 12 fit and 4 validation queries after seed-42 query-disjoint
  split of official SciFact train; first 20 BM25 candidates per query.
- Record: number of retrieved positives, student/BM25 same-subset nDCG@10 and
  MRR@10, encoder/cache wall time, head fit time, model and library versions.
- **Not comparable** to 7B–7E 162-query, top-100 validation metrics.
  Previously reused validation is exploratory; no official test access.

## 8A frozen semantic encoder feasibility (measured)

- Commit: `76f78a3`; Intel Mac CPU, Python 3.10.15.
- Official SciFact train-only subset: 12 fit queries / 240 pairs; 4 validation
  queries / 80 pairs (BM25 top-20); all four validation queries had a retrieved
  positive, with just four judged and 76 unjudged validation candidates.
- Frozen `sentence-transformers/all-MiniLM-L6-v2`; 10-parameter hybrid head.
- Text head and BM25 both achieved MRR@10 **1.0**, nDCG@10 **1.0** on the
  same four queries. **No demonstrated ranking improvement.**
- First embedding stage **26.0961 s**, cached rerun **0.1422 s**;
  head fit **0.0526 s** (first run). These timings are not general latency estimates.
- Feasibility only: tiny, easy, reused validation subset; not comparable to
  the 162-query top-100 results; no official test data used.

## 8B scaled frozen semantic student (planned; measurements TBD)

- Full official SciFact train, 647 fit / 162 validation, BM25 top-100.
- Compare rich 8-feature linear, rich MLP-8, and frozen MiniLM embedding
  10-feature linear head, all hybrid (T=2, hard weight 0.5, 120 steps, seed 42).
- Teacher scores must be fully cached. SQLite embedding checkpoints are
  resumable; log unique text count, cache hits, CPU runtime, and ranking metrics.
- **TBD:** measured model metrics and runtimes; do not invent results.
- Previously reused validation remains exploratory. No official test access.

## 8B scaled frozen semantic student (measured)

- Commit: `40db20b`; Intel Mac CPU, Python 3.10.15; SciFact official train only,
  647 fit / 162 validation queries, BM25 top-100, 64,700 fit / 16,200 validation pairs.
- Frozen `sentence-transformers/all-MiniLM-L6-v2`, 384-dimensional embeddings;
  5,979 unique texts, first cache fill **280.0418 s**, fully cached read **0.3914 s**.
- All-query nDCG@10 / MRR@10: linear **0.6828966118 / 0.6496472663**;
  MLP-8 **0.6841569805 / 0.6509577699**;
  BM25 **0.6883649753 / 0.6522584754**;
  semantic 10-feature head **0.6987394572 / 0.6622232020**.
- Semantic head training **2.8610 s**, feature construction **44.9807 s**
  (first run); head has 10 trainable parameters, encoder frozen.
- Validation: 150/162 queries have a retrieved positive; 170 judged and
  16,030 unjudged candidate pairs. Semantic vs BM25 delta nDCG@10 **+0.0103744819**.
- Exploratory, repeatedly reused validation; no paired significance result
  yet, no official test used. Fixed-pool reranking, not full-corpus retrieval.

## 8C semantic feature ablation (planned; measurements TBD)

- Same official SciFact train-only 647/162 query split, BM25 top-100, cached
  teacher and frozen embeddings, seed 42, 120 steps, hybrid objective.
- Compare lexical-only (8 active features), semantic-only (2), combined (10),
  plus BM25 and MLP-8 references; identical query candidate pools.
- Paired query bootstrap CIs and sign-flip tests (10,000 replicates, seed 42),
  including combined vs BM25 and MLP-8. Multiple exploratory comparisons.
- **TBD:** actual metrics, deltas, confidence intervals, p-values, and CPU timings.
- No official test data used; validation repeatedly reused; no claims of
  confirmatory significance or generalization from this one split.

## 8C semantic feature ablation (measured)

- Commit `7cf2fb6`; 302 tests passed; official SciFact train split only,
  647 fit / 162 validation queries, BM25 top-100, frozen MiniLM encoder.
- All-query validation nDCG@10 / MRR@10: lexical-only **0.6828966118 /
  0.6496472663**; semantic-only **0.7029141406 / 0.6707206545**;
  combined **0.6987394572 / 0.6622232020**; BM25 **0.6883649753 /
  0.6522584754**; MLP-8 **0.6841569805 / 0.6509577699**.
- Combined minus BM25 nDCG delta **+0.01037448**, 95% query-bootstrap CI
  **[0.00155209, 0.01944628]**, unadjusted paired sign-flip p **0.01870**.
- Combined minus semantic-only nDCG delta **-0.00417468**, CI
  **[-0.05208216, 0.04440079]**, p **0.86971**. Semantic-only minus
  lexical-only nDCG delta **+0.02001753**, CI **[-0.03122253, 0.07172332]**,
  p **0.45865**. Large uncertainty; do not claim reliable superiority.
- All 5,979 cached embeddings reused; 0 new. **170 judged / 16,030
  unjudged** validation candidates. Multiple unadjusted comparisons and
  repeatedly reused validation: **exploratory only**; no official test used.

## 8D evaluation audit (planned; measurements TBD)

- Replay the fixed 8C model configuration without any teacher or encoder
  inference; fail on incomplete caches.
- Require exact score/candidate alignment, record order-sensitive pool
  fingerprint, and reconcile per-query means with aggregate metrics.
- Compare `nonrelevant` and `judged_only` sensitivity scenarios, plus query
  delta distributions and spread to investigate wide 8C intervals.
- **TBD:** actual audit outputs and interpretation from Intel Mac run.
- Neither judgment policy recovers missing relevance labels. Reused
  validation remains exploratory; official test data prohibited.

## 8D fixed-pool audit (measured)

- Commit `f6c9e98`; 316 tests passed. Official train-only 647/162 validation;
  all 5,979 cached embeddings reused, zero new inference.
- Validation pool SHA256 `efca1a9fa6463a831749c747b9a3d1543f18161c474aadec63750949aa6cc440`.
  Per-query and aggregate MRR/nDCG reconciled for all audited comparisons.
- Nonrelevant-policy nDCG: combined vs BM25 **+0.01037448** (21 improved,
  6 regressed; query-delta SD **0.05794**); semantic-only vs BM25
  **+0.01454917** (38 improved, 33 regressed; SD **0.31701**);
  combined vs semantic-only **-0.00417468** (34 improved, 36 regressed;
  SD **0.31759**). Remaining queries unchanged.
- All judged-only pair differences were **zero** because removing unjudged
  passages leaves too little judged ranking variation; **not equivalence**.
  170 judged and 16,030 unjudged validation candidates. This does not
  resolve incomplete qrels or establish generalization.

## 8E query-level error analysis (planned; measurements TBD)

- Replay 8C semantic-only and combined heads, compare with original BM25
  candidate ordering on the same validation SHA256; forbid new inference.
- Export deterministic per-query gains/losses and top-five cases; stratify
  descriptively by query token count, overlap with original BM25 top-1,
  and BM25 rank of the best judged relevant candidate.
- **TBD:** actual subgroup counts, means, extreme cases and interpretation.
- Post-hoc descriptive diagnostics only; reused validation, incomplete qrels,
  and no official test data.

## 8E query-level diagnostics (measured)

- Commit `484c2cc`; 326 tests passed; train-only validation fingerprint
  `efca1a9fa6463a831749c747b9a3d1543f18161c474aadec63750949aa6cc440`.
- Semantic-only minus BM25: mean nDCG@10 **+0.01454917** across 162 queries;
  **38 improved, 33 regressed, 91 unchanged**.
- Of 135 queries with a judged positive originally in BM25 top 10, mean
  delta **-0.02556136** (29 improved, 33 regressed). Of 15 with a positive
  at BM25 ranks 11–100, mean delta **+0.38718318** (9 improved, 0 regressed).
  Twelve queries had no retrieved judged positive, delta zero.
- Semantic-only minus combined: **+0.00417468** mean nDCG, 36 improved,
  34 regressed, 92 unchanged. These post-hoc bins are descriptive only;
  sparse judgments and reused validation preclude confirmatory inference.

## 9A matched-supervision controls (planned; measurements TBD)

- Freeze encoder, candidate pool, split, feature block, initialization and
  optimization. Compare soft teacher-only, hard relevance-only, hybrid and
  (semantic block) untrained cosine, with BM25 reference.
- Train-only 647/162; cache-only replay; paired query statistics; 8D pool
  fingerprint. **TBD:** actual scores, deltas and uncertainty after Mac run.
- Primary scientific question: Does teacher supervision help over the same
  architecture trained without teacher targets? Reused validation means
  results are exploratory, not confirmatory.

## 9A controlled supervision (measured)

- Commit `358ec88`; **338 tests passed**; train-only 647 fit / 162 validation;
  fingerprint `efca1a9fa6463a831749c747b9a3d1543f18161c474aadec63750949aa6cc440`.
- Semantic-only (2 active features): teacher-only, relevance-only, hybrid,
  and untrained cosine all **MRR@10 0.67072065 / nDCG@10 0.70291414**;
  zero observed pairwise metric differences. Identical metrics do not prove
  identical learned weights, scores, or full candidate ordering.
- Combined (10 features): relevance-only nDCG **0.69720235**; teacher-only
  **0.69807287**; hybrid **0.69873946**. Hybrid minus relevance-only
  **+0.00153711** nDCG, bootstrap 95% CI **[-0.00027052, 0.00388285]**,
  sign-flip **p=0.17848**. Teacher-only minus relevance-only **+0.00087051**,
  95% CI **[-0.00191306, 0.00369559]**, **p=0.56754**.
- No statistically established teacher-supervision advantage. Results remain
  exploratory: reused validation, incomplete judgments, unadjusted tests.

## 9B optimization and ranking diagnostics (planned; measurements TBD)

- Compare matched objectives with trace checkpoints, final active head
  coefficients, fit-query teacher rank agreement, and validation ordering
  changes from the same random initialization.
- **TBD:** measured trajectories, order changes, agreement and interpretation.
- Teacher agreement is **in-sample on fit queries** only, not validation
  teacher agreement or a claim of generalization.

## 9B learning diagnostics (measured)

- Commit `258f5a7`; **347 tests passed**; 647 fit / 162 exploratory validation.
- Semantic-only teacher-only soft KL **1.90593 → 1.65636**; hybrid **1.90593 → 1.68864**;
  relevance-only **1.90593 → 1.72127**. Learned cosine coefficients 2.44655,
  2.11710, 1.78830 respectively; identical validation ranking metrics in 9A.
- Combined teacher-only soft KL **1.90619 → 0.69545**; hybrid **1.90619 → 0.83653**;
  relevance-only **1.90619 → 0.98125**. Fit-query teacher top-1 agreement
  ~0.623, 0.623, 0.620 respectively. **In-sample**, not generalization.
- Validation ordering changes are versus zero-centered initialized head, **not**
  raw cosine or BM25; zero-initialization ties explain the 162/162 changes in
  semantic-only heads. Optimization works, but relevance gains remain unproven.

## 9C teacher–relevance alignment audit (planned; measurements TBD)

- **TBD:** teacher/BM25 fit-query nDCG@10, per-query disagreements,
  best judged-positive ranks, teacher top-1 judgment status and margins.
- Distinguish unjudged passages from judged negatives. Only report margins to
  *verified* judged zeros when those exist; also report separately the margin
  against the highest-scoring nonpositive candidate, which may be unjudged.
- All 9C findings are descriptive and **in-sample on 647 fitting queries**.
  No teacher inference, new embeddings, validation teacher inference or test use.

## 9C teacher–relevance alignment (measured)

- Commit `d5cc054`; **354 tests passed**; 647 fit queries only; no validation
  or official test evaluation.
- BM25 nDCG@10 **0.66727285**, teacher **0.69353798** (difference **+0.02626513**);
  140 improved / 87 regressed / 420 unchanged.
- Teacher top-1: 371 judged relevant, **276 unjudged**; no verified judged-zero
  margins available. An unjudged teacher choice is not a proven error.
- By original BM25 judged-positive rank: rank 1 (363 queries) Δ **−0.05985776**;
  ranks 2–10 (156) Δ **+0.14290813**; ranks 11–100 (68) Δ **+0.24159176**;
  no retrieved positive (60) Δ **0**. Qrel-derived bins are **not**
  inference-time routing features. Descriptive in-sample analysis only.

## 9D confidence-weighted teacher supervision (planned; measurements TBD)

- Match the combined ten-feature student and 9A hybrid optimization; vary
  only per-query weights on the soft teacher gradient, with mean-one normalization.
- Uniform, bounded teacher top-two margin, teacher/BM25 top-1 agreement,
  and deterministic shuffled-margin negative control. No relevance labels
  used to compute weights. Keep hard-label gradients unchanged.
- **TBD:** validation MRR/nDCG, paired deltas, CIs, p-values and weight
  distribution. Validation is reused and exploratory; no official test.
