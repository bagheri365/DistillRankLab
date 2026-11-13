# Milestone 9C — Teacher–relevance alignment audit

**Research question:** Does the frozen cross-encoder reliably elevate judged
relevant SciFact passages within BM25's fixed candidate pool? This audit helps
interpret why 9B teacher imitation did not yield meaningful relevance gains.

## Run (CPU, cache only)

```bash
pytest
python examples/26_scifact_teacher_alignment.py \
  --train-data datasets/scifact_full_train_bm25 \
  --teacher-cache datasets/scifact_full_train_teacher_scores.json \
  --output datasets/scifact_9c_teacher_alignment.json
```

This runs on **647 fitting queries only**, checks the 162-query validation
fingerprint without scoring validation queries, and refuses missing teacher
scores. No encoder, downloads, new teacher inference or official test access.

The JSON report contains full per-query diagnostics and the terminal displays
summary statistics, subgroup counts and the five largest gains/losses. The
script reconstructs BM25 **rank scores** from the original candidate order;
original numerical BM25 scores are not available.

**Interpretation:** A teacher top-1 candidate without a judgment is **unknown**,
not necessarily wrong. A margin to the highest-scoring other candidate is not
a verified positive-vs-negative margin. Sparse SciFact judgments, training-set
measurement, and post-hoc bins preclude confirmatory claims. This audit does
not evaluate held-out teacher generalization or the effect of distillation.
