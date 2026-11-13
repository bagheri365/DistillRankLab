# DistillRankLab research protocol

## Aim and status

Evaluate whether a small, CPU-friendly student can approximate a frozen
cross-encoder for reranking scientific claims over BEIR SciFact. Milestones
6D–6G established the data, candidate pool, teacher baseline and paired
uncertainty estimates. Milestone 7A introduced the first four-feature linear
student. Milestone 7B adds **train-only validation selection**.

## Dataset and experimental units

- Official SciFact train and test splits; each query is an independent unit
  for train/validation splitting and paired statistical resampling.
- BM25 top-100 candidates are retrieved without positive injection; document
  relevance judgments do not influence candidate generation.
- All-query metrics count queries without retrieved judged positives as zero.
  Conditional metrics exclude those queries and must be labeled separately.
- Unjudged candidates are **not verified negatives**. The default evaluation
  treats them as nonrelevant, and hard-label training makes the same assumption.
- Metrics are for a **fixed BM25 candidate pool**, not the official full-corpus
  BEIR evaluation.

## Previously observed test results (exploratory)

Full SciFact test: 300 queries, 30,000 BM25 top-100 candidates, 268 queries
with retrieved judged positives. All-query BM25 MRR@10 0.6276 and nDCG@10
0.6639; frozen cross-encoder MRR@10 0.6509 and nDCG@10 0.6809. Paired
10,000-replicate tests yielded MRR difference +0.02330 (95% bootstrap CI
[-0.01048, 0.05773], sign-flip p=0.1764) and nDCG difference +0.01704
(95% bootstrap CI [-0.01247, 0.04722], p=0.2675). These are **not**
statistically significant at 0.05. The 7A soft student scored test all-query
nDCG@10 0.64615; hybrid 0.64617. Those test results were examined before 7B,
so the 7B evaluation remains exploratory, not an untouched confirmatory test.

## 7B predeclared selection protocol

Use official SciFact **train** query groups only. Split 80/20 by stable
SHA-256 query ID with seed 42. Select maximum validation **all-query
nDCG@10** over the explicitly recorded hard/soft/hybrid grid; break ties by
first configuration. Refit chosen configuration on the entire official train
split. Only after selection, optionally evaluate on official test. No teacher
scores or labels from test may enter fitting, validation, or selection. All
trials must use the same train/validation split and first-stage candidate pool.

See `examples/14_scifact_validation_README.md` for exact commands and default
grid. Generated reports and cached scores belong in ignored `datasets/`.

## Claims and limitations

A larger observed score does not alone establish statistical superiority.
Document the selection process and the number of tried configurations. The
four-feature linear model is an interpretable baseline, not a neural encoder;
any performance gap may reflect features, model capacity, optimization, or
supervision assumptions. Never describe the 7B test run as a fresh untouched
confirmatory evaluation, given the exploratory 7A test results.

## 7C feature ablation and registry

Milestone 7B selected a four-feature soft student (temperature 2) with
validation all-query nDCG@10 **0.682475**, versus BM25 **0.688365** on the
same 162 validation queries. Its refit test score was **0.644325** nDCG@10
(exploratory; the test split was previously examined).

Milestone 7C compares baseline (4), lexical (6), and rich (8) feature sets,
each with hard, soft and hybrid objectives, on the **identical train-only
validation split**. Optimization settings are fixed across feature sets;
no official test data enters selection or reporting. See
`examples/15_scifact_feature_ablation_README.md` for definitions and commands
and `docs/experiment_registry.md` for observed outcomes. The extra features
are deterministic within each query-document pair; BM25 rank is retained.
Validation reuse and previous model comparisons make 7C **exploratory**, not
confirmatory. Any performance change may reflect features, dimensionality,
or optimization interactions; it does not establish a causal explanation.

## Milestone 7D: controlled nonlinear capacity ablation

Use the **same official-train-only** query-disjoint validation split (seed 42)
and the same eight rich features for a linear model and two small tanh MLPs
(8 and 16 hidden units). Keep the listwise hybrid objective, temperature,
learning rate, steps, seed, and fixed BM25 candidate pool identical. Compare
all-query nDCG@10 and MRR@10, trainable parameter counts, and measured runtime.
Select on validation nDCG only and refit on official training queries. Do not
use official test scores for capacity selection. This split was used in 7B/7C,
so 7D validation results are exploratory, not an independent confirmation.
See `examples/16_scifact_nonlinear_ablation_README.md`.

## 7E multi-seed robustness protocol

Repeat the predeclared 7D rich-feature hybrid linear, MLP-8 and MLP-16
configurations across five distinct training seeds on the same train-only
647/162 query split. Report mean/sample SD of validation all-query MRR@10
and nDCG@10 across seeds, and per-query paired uncertainty conditional on the
evaluated seeds. No official test split, model selection or test-based tuning.
This reused validation is exploratory; seed variation and query uncertainty
are distinct, and multiple comparisons are uncorrected. See
`examples/17_scifact_multiseed_README.md`.

## 7F reproducible consolidation

Use `python examples/18_scifact_research_summary.py --input datasets/scifact_multiseed.json`
to generate a machine-readable and Markdown summary from the full 7E report.
The summary validates the **five predeclared seeds**, 120 training steps,
647/162 query counts, and the absence of test use, rejecting the smoke test.
It does not train, resample, load SciFact test data, or re-run the teacher.
It includes the reported paired intervals and p-values, which are
conditional on evaluated training seeds, uncorrected for multiple tests,
and exploratory because the validation split was repeatedly inspected.
The legacy `mean_teacher_minus_bm25` statistic key remains available for
compatibility; 7E also exposes `mean_candidate_minus_reference` to clarify
that the comparison is MLP minus linear, not teacher minus BM25.

## 8A: frozen pretrained text features (feasibility only)

The initial neural-representation study uses a frozen pretrained sentence
encoder to embed queries and passages on CPU. Its two embedding interactions
are concatenated with the established eight rich lexical features, and a
10-weight linear listwise head is trained with cached teacher scores. This
isolates pretrained semantic representation from head capacity; it is **not**
a trained text encoder or a cross-encoder student. Teacher inference and
official test access are disallowed by the experiment script.

The deliberately small, BM25-truncated train-only subset makes runtime
manageable but changes retrieval coverage and evaluation difficulty. Only
compare the student against BM25 **on exactly the same subset**. Do not
interpret a subset difference as a benchmark improvement, nor compare its
absolute scores to 7E. See `examples/19_scifact_text_feasibility_README.md`.

## 8B: scaled frozen semantic representations

Keep the official SciFact test split excluded. Use the same seed-42 query-disjoint
647/162 official-train split and the original BM25 top-100 candidate pool for
all three models. The baselines are rich linear and MLP-8 students; the text
student uses a frozen MiniLM encoder with two additional embedding interaction
features and a trained linear head. Keep hybrid objective, seed, and optimization
settings fixed. Do not claim a statistically significant improvement from the
reused validation split. The encoder is **not** fine-tuned. Use checkpointed,
model-ID-bound SQLite embedding storage; report cold-cache versus warm-cache
costs separately. No new cross-encoder inference is allowed.

## 8C frozen semantic-feature block ablation

Hold official SciFact test data out of this stage. Use the fixed query-disjoint
647/162 split of the official train pool (BM25 top-100, seed 42). Reuse the
frozen MiniLM embeddings and cached cross-encoder scores. Compare a shared
10-coordinate linear ranking head with lexical-only (8 active coordinates),
semantic-only (2), and combined (10). Mask inactive columns to zero before
fitting and scoring; preserve optimizer, initialization seed, objective and
steps. Report MRR@10 and nDCG@10 on all 162 validation queries, including
zero-credit queries with no retrieved positive. Compare the combined head to
BM25 and MLP-8 using query-paired bootstrap confidence intervals and sign-flip
p-values; compare feature blocks likewise. Report 10,000 resamples, seed 42,
and explicitly label all five comparisons exploratory and unadjusted for
multiplicity. Query resampling is not training-seed uncertainty. The validation
split has already been inspected repeatedly and is not an untouched holdout.
Record all results (including null findings) in the experiment registry.

## 8D audit protocol

Before any further model selection or encoder fine-tuning, replay the fixed
8C configuration on the **train-only** 647/162 split. Refuse incomplete
teacher/embedding caches, verify exact candidate-score keys, and record a
candidate-order and qrel-status SHA256 fingerprint. Check all-query aggregate
MRR/nDCG against the mean of 162 query-level values for each ranker and each
unjudged policy. Report distributional spread (not only means) of paired
query differences, and compare `nonrelevant` with `judged_only` as **distinct
sensitivity scenarios**. Judged-only removes unjudged passages from rankings;
it is not a fix for missing judgments. Do not use official test data or
present exploratory validation comparisons as confirmatory findings.

## 8E query error protocol

Use the exact 8D validation-pool fingerprint and train-only 647/162 split.
Recreate the 8C fixed-seed semantic-only and combined linear heads from
cached teacher scores and frozen text embeddings. No new model inference,
no official test data. Compare semantic-only with BM25 and combined using
per-query MRR@10 and nDCG@10 under the nonrelevant policy. Report query IDs,
original BM25 positive rank, query length, lexical overlap with the original
BM25 top-1 candidate, top-1 passage IDs, and the largest gains/losses.
Aggregate descriptive bins with counts and mean deltas; do not treat
post-hoc subgroups as statistically validated or causal explanations.
Record every query, including zero-positive-coverage queries. Preserve
missing judgments as unjudged metadata and do not claim a zero-gain passage
is truly irrelevant. The validation split has been reused repeatedly.

## 9A controlled supervision protocol

Hold the frozen `all-MiniLM-L6-v2` encoder, 8D audited candidate pool, 647/162
query split, feature matrices, seed 42, 120 optimizer steps and learning rate
0.05 constant. For each of semantic-only (2 active coordinates) and combined
(10 active), fit the identical head with hard weights 0 (teacher-only), 0.5
(hybrid), and 1 (relevance-only). Add a no-training raw embedding-cosine
baseline for semantic-only and the original BM25 ranking. Compare teacher-only
and hybrid against relevance-only, and hybrid against teacher-only, with paired
query bootstrap and sign-flip diagnostics (10,000 replicates). Teacher-only
must not use hard-label gradients; relevance-only must not use teacher gradients.
No encoder/teacher inference or official test data. Verify exact score-pool
alignment and the fixed SHA256. Record the full report even when results are
null or negative. All inference remains exploratory due to repeatedly reused
validation, sparse judgments, and unadjusted multiple tests. This isolates
supervision conditional on a pretrained encoder, not the origin of the
encoder's pretrained knowledge; no end-to-end CPU latency claims.

## 9B learning diagnostics protocol

Use the same 647/162 train-only split, 8D validation fingerprint, BM25 top-100,
model-bound embedding cache and frozen teacher-score cache as 9A. Fit identical
semantic-only and combined linear heads under teacher-only, relevance-only,
and hybrid supervision, seed 42, 120 steps, learning rate 0.05 and temperature
2. Record soft KL and hard-label cross-entropy at steps 0, 1, 10, 30, 60, 120
(omit checkpoints exceeding the configured step count), even when a loss term
has zero training weight. Check that instrumented and uninstrumented fits
produce identical coefficients. Report active weights and fit-query teacher
agreement (top-1 and stable-tie Spearman), explicitly marked **in-sample**.
Count exact validation top-1, top-10 order, full order and top-10 membership
changes versus the identical untrained initialization. These diagnostics
measure optimization and ranking behavior, **not** statistical significance,
causality, or held-out teacher imitation. No new model inference or official
test data. Keep measured findings TBD until a real Mac experiment is run.

## 9C teacher–relevance alignment protocol

Use the full 809-query official SciFact TRAIN BM25 top-100 candidate export,
SHA256-audited 647/162 split (seed 42), and existing frozen cross-encoder
teacher cache. Audit **only 647 fit queries**. Fail closed if the cache is
incomplete; never run model inference, query validation teacher scores or use
the official test set. Preserve the BM25 candidate order, fixed pool and
unjudged-vs-judged status. Report teacher/BM25 nDCG@10 and MRR@10, paired
per-query gains and losses, teacher top-1 judgments (relevant, judged zero,
unjudged), best judged-positive rank under each ranking, and top-1 disagreement.
For score margins distinguish (1) best judged positive vs highest-scoring
*other* candidate, often unjudged, from (2) best judged positive vs strongest
*verified judged-zero* candidate, only when both labels exist. The former is
not a positive/negative classification margin; unjudged top-1 is not a
teacher error. Group by initial BM25 positive rank and disagreement status;
report bin sample counts and example query IDs. No statistical significance
claims from exploratory post-hoc subgroups. Keep findings TBD until measured.

## 9D confidence-aware supervision protocol

On the existing SHA256-audited 647 fit / 162 exploratory validation split,
train four otherwise identical combined ten-feature linear heads (seed 42,
120 steps, learning rate .05, temperature 2, hard weight .5). Use frozen cached
MiniLM embeddings and cross-encoder scores only; fail closed on cache misses.
Apply mean-one per-query weights **only to soft teacher gradients**:
uniform (1); bounded top-1 minus top-2 teacher logit gap
`0.5 + gap/(1+gap)`; top-1 teacher/BM25 agreement (1.5 vs .5); and a seeded
permutation of margin weights. Compute weights without qrels, never use
validation judgments to choose thresholds, and keep hard gradients identical.
Check that uniform weighting reproduces the original hybrid head exactly.
Report all-query nDCG@10/MRR@10, query-paired uncertainty versus uniform,
and weight ranges. Do not treat a teacher score gap as calibrated confidence,
a single shuffled run as a significance test, or repeated validation as
confirmatory evidence. No official test access.
