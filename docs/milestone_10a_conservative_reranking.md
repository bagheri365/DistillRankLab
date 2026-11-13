# Milestone 10A — conservative semantic reranking

## Prior 9D measured findings


- Commit `c9ee11f`; **365 tests passed**; 647 fit / 162 reused exploratory validation.
- nDCG@10: uniform **0.69873946**, shuffled-margin **0.69873946**,
  margin-weighted **0.69750333**, agreement-weighted **0.69728532**;
  BM25 **0.68836498**. No confidence rule improved on uniform.
- Paired nDCG versus uniform: margin **−0.00123613** (sign-flip p=0.503),
  agreement **−0.00145413** (p=0.125), shuffled **0** (p=1).
  Bootstrap/sign-flip disagreement and repeated validation use preclude
  confirmatory conclusions. One shuffled control is not a null distribution.
- No official test used; sparse judgments and encoder inference costs remain
  unresolved. **No reliable incremental distillation gain established.**


## 10A planned experiment


- **TBD:** frozen MiniLM cosine, BM25 rank-order baseline, rank blends with
  alpha 0.25/0.50/0.75, and top-1 guarded variants (cosine gap <= 0.05).
- Report paired deltas against **both** BM25 and raw cosine and regression
  counts. No qrels or teacher scores enter any reranking rule.
- Reuse the audited 162-query validation pool, cache-only embeddings, no test.
  This is exploratory **ranking-system design**, not distillation evidence.

## 10A research protocol


Use BM25's original top-100 candidate order (not numerical BM25 scores),
frozen MiniLM cosine similarities, and normalized rank-space convex blends
with predeclared alpha values 0.25, 0.50, and 0.75. For each blend, evaluate
a guarded variant that retains the BM25 top-1 passage if the semantic winner's
cosine advantage over that passage is <= 0.05. This guard depends only on
scores and retrieval order, never on qrels. Evaluate all eight rankers on the
same audited 162-query exploratory validation pool. Include per-query paired
uncertainty and regression counts against both standalone baselines. Keep the
entire grid visible; do not present its best observed variant as independently
selected. Sparse judgments, reused validation, lack of end-to-end latency,
and absence of a distillation mechanism remain explicit limitations.
