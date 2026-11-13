# Milestone 10B — frozen-shortlist robustness and stability audit

## Measured 10A findings (exploratory, 162 reused SciFact validation queries)

- Commit `e52ba17`, 377 Mac tests passed; no official test used.
- All-query nDCG@10 / MRR@10: BM25 **0.68836498 / 0.65225848**;
  frozen cosine **0.70291414 / 0.67072065**;
  50% rank blend **0.71801358 / 0.68545463**;
  75% rank blend **0.71990423 / 0.69265873**.
- Guarded 75% blend fell to nDCG **0.70711630**: top-1 protection
  was not beneficial in the observed aggregate.
- 75% blend vs BM25 nDCG delta **+0.03153926** (95% query-bootstrap
  interval **[-0.00885533, +0.07113878]**, sign-flip p **0.1325**).
  Against cosine delta **+0.01699009** (interval **[-0.00759524,
  +0.03957085]**, p **0.1645**). Neither comparison establishes
  a reliable improvement. Multiple comparisons and repeated validation
  prohibit confirmatory interpretation.

## 10B protocol

Freeze the shortlist to **BM25**, **cosine**, **blend_0.5**, and
**blend_0.75**, selected after 10A's exploratory results. No further
blend-weight or guard-threshold tuning. On the identical 162-query
validation pool, report:

1. nDCG@10 and MRR@10 (all-query and conditional), per-query paired
   gains/losses versus BM25 and cosine, and top five gain/loss examples.
2. Top-1 agreement and mean top-10 candidate overlap with BM25.
3. Best judged positive rank transitions from BM25 to 75% blend, as
   **post-hoc diagnostics only**. Never route using qrels at inference.
4. Leave-one-query-out mean delta ranges, to expose sensitivity to
   individual queries; these are **not** confidence intervals.
5. Candidate-pool truncation to original BM25 top-20/top-50/top-100,
   recomputing rank blends and reporting positive-retrieval coverage.
   Truncation changes the retrieval ceiling, so cross-cutoff comparisons
   do not isolate ranking robustness.

Use frozen `sentence-transformers/all-MiniLM-L6-v2` embeddings from the
existing SQLite cache only, with an encoder callback that raises if
uncached inference is requested. Enforce the audited validation pool
SHA256 `efca1a9fa6463a831749c747b9a3d1543f18161c474aadec63750949aa6cc440`.
Do not train or score the official SciFact test set. Full JSON contains
per-query diagnostics; stdout prints a compact summary. This is
**ranking-system analysis**, not distillation evidence.

## Limitations and interpretation

The 10B shortlist is not independently selected, and the validation split
has been examined repeatedly. Most retrieved candidates are unjudged and
are treated as nonrelevant by the benchmark's metrics. Rank changes and
apparent losses may reflect missing judgments. Cached embedding retrieval
is not end-to-end CPU inference latency. An independent benchmark is
required for confirmatory ranking claims.

## 10B results

**TBD — populate after running `examples/29_scifact_reranking_robustness.py`.**
