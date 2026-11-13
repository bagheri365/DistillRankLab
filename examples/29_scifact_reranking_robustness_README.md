# 10B: frozen-shortlist reranking robustness

Run after 10A, with the same SciFact training BM25 top-100 export and
complete MiniLM embedding SQLite cache:

```bash
python examples/29_scifact_reranking_robustness.py \
  --train-data datasets/scifact_full_train_bm25 \
  --embedding-cache datasets/scifact_8b_embeddings.sqlite \
  --output datasets/scifact_10b_reranking_robustness.json
```

The script prints a compact summary and saves all per-query details to
JSON. It runs no teacher or encoder inference, does not train, and does
not evaluate the official test set. See
`docs/milestone_10b_reranking_robustness.md` for the frozen protocol and
limitations. Report all four methods and all three pool cutoffs; do not
select a new hyperparameter from these results.
