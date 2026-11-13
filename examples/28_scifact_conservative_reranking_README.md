# Milestone 10A — Conservative semantic reranking

This exploratory, **non-distillation** experiment compares the original BM25 top-100
order, frozen MiniLM cosine ranking, and predeclared convex rank blends
(alpha = 0.25, 0.50, 0.75). Each blend also has a guarded version that
preserves the original BM25 top-ranked passage **only if** the semantic winner's
cosine advantage over that passage is at most 0.05. This is a deliberately
simple, inference-available heuristic, **not** an oracle for relevance.

```bash
python examples/28_scifact_conservative_reranking.py \
  --train-data datasets/scifact_full_train_bm25 \
  --embedding-cache datasets/scifact_8b_embeddings.sqlite \
  --output datasets/scifact_10a_conservative_reranking.json \
  --replicates 10000
```

Requires the full official SciFact **train** BM25 top-100 preparation and a
complete frozen-encoder embedding cache. Splits by query using the original
seed-42 647/162 partition; only the 162 exploratory validation queries are
scored. Fingerprint must match the audited 8D pool. Missing embeddings cause
an error; no model inference, teacher scores, or official test data are used.

The output includes all-query MRR@10/nDCG@10, paired bootstrap intervals and
sign-flip tests versus both BM25 and raw cosine, plus the number of queries
with nDCG regressions relative to each baseline. The BM25 signal is its
**original rank order**, not unavailable numerical BM25 scores. Cosine is
rank-normalized for blending. Guarding can preserve bad BM25 decisions too.

The validation set has been repeatedly inspected; all results and paired tests
are **exploratory**, with unjudged passages treated as nonrelevant and multiple
unadjusted comparisons. Do not select a final model based on these numbers
and then claim confirmatory performance on the same validation data.

This experiment does not measure end-to-end inference cost or demonstrate any
incremental benefit from knowledge distillation.
