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
