# Milestone 8E — query-level error analysis

Run on the **official SciFact train pool only**, with the existing SHA256
647-fit/162-validation split and unchanged BM25 top-100 candidate order.
No new teacher or encoder inference is allowed. Both caches must be complete.

```bash
python examples/23_scifact_query_errors.py \
  --train-data datasets/scifact_full_train_bm25 \
  --teacher-cache datasets/scifact_full_train_teacher_scores.json \
  --embedding-cache datasets/scifact_8b_embeddings.sqlite \
  --output datasets/scifact_8e_query_errors.json
```

The JSON records all 162 per-query nDCG/MRR differences for semantic-only
versus BM25 and versus combined, plus the top five gains/losses. The terminal
prints a compact subset. Query IDs, top-1 passage IDs, token counts, lexical
overlap between query and original BM25 top-1 passage, and BM25 rank of the
best retrieved judged positive aid inspection. The query text and passage
text are not exported. The three bins for query length, top-1 overlap and
best relevant BM25 rank are **predeclared descriptive bins**, not causal
subgroups or statistically tested discoveries.

The `best_relevant_bm25_rank` uses the candidate TSV's original order,
not numerical BM25 scores. `not_retrieved` means no *judged* positive in the
100 candidates, not necessarily that no relevant passage exists. A model's
top-1 candidate may be unjudged. Unjudged passages are counted as zero gain
for nDCG/MRR but not claimed to be truly irrelevant. These exploratory
results come from a repeatedly reused validation split and cannot confirm
model superiority or guide a confirmatory test claim.
