# 9D — Confidence-aware distillation (exploratory)

Use the existing full official SciFact TRAIN BM25 top-100 export, frozen cross-encoder
teacher cache, and SQLite frozen MiniLM embedding cache. No model downloads,
new inference, or official test evaluation are allowed. The script fails closed
if either cache is incomplete or the validation fingerprint differs.

```bash
python examples/27_scifact_confidence_distillation.py \
  --train-data datasets/scifact_full_train_bm25 \
  --teacher-cache datasets/scifact_full_train_teacher_scores.json \
  --embedding-cache datasets/scifact_8b_embeddings.sqlite \
  --output datasets/scifact_9d_confidence_distillation.json \
  --replicates 10000
```

The **same ten-feature linear student** and frozen encoder are trained for
120 steps, seed 42, temperature 2, learning rate .05, hard weight .5.
Only the *soft-teacher gradient* changes: uniform, teacher top-two score gap,
teacher/BM25 top-1 agreement, or shuffled margin weights. Every weight vector
has mean one. Hard-label gradient is unchanged. Teacher margin and agreement
are proxies for confidence, **not verified correctness**. Shuffled weights
preserve the margin-weight distribution but break query correspondence;
one permutation is not a permutation significance test.

Results report all-query MRR@10 and nDCG@10, paired query-level bootstrap
intervals and sign-flip p-values against uniform. All results are exploratory:
162 repeatedly inspected validation queries, sparse qrels (unjudged treated
nonrelevant), unadjusted multiple comparisons, and no confirmatory test.
