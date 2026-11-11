# Milestone 6F: full SciFact test split and query-level diagnostics

The `--max-queries` flag is now **optional**; omitting it selects every query
with a qrel in the official SciFact test split (rather than the previous default
of 20). Use a new output directory so the five-query baseline remains intact.

```bash
python examples/09_prepare_scifact.py --download \
  --split test --top-k 100 --output datasets/scifact_full_test_bm25
```

Check `manifest.json` for the actual query count, candidate count and
`queries_with_retrieved_positive`. The candidate pool is produced by BM25
without looking at qrels. This is not a full-corpus reranking experiment.

Run frozen cross-encoder scoring on CPU (potentially several thousand pairs;
this may take minutes or longer). Keep the cache for restart/reuse:

```bash
python examples/10_scifact_reranking.py \
  --data datasets/scifact_full_test_bm25 \
  --cache datasets/scifact_full_teacher_scores.json \
  --batch-size 8 --k 10 --unjudged nonrelevant \
  --output datasets/scifact_full_results.json \
  --per-query-output datasets/scifact_full_per_query.json
```

The summary includes conditional and all-query MRR@10 and nDCG@10, retrieval
coverage, judgment coverage and timing. Per-query JSON records both rankers'
MRR/nDCG, signed deltas, and nDCG improved/regressed/unchanged counts. Queries
with no retrieved positive count as zero in all-query means and cannot be
recovered by reranking. `nonrelevant` treats unjudged documents as zero-gain,
**not** as verified irrelevant; `judged_only` is an alternate evaluation policy.

The BM25 baseline uses the original candidate order (rank scores, not original
BM25 numerical scores). Scores for the cross-encoder are cached by text/model.
The reported scoring time excludes model loading and includes cache I/O; a warm
cache run is not a valid neural inference timing benchmark. These are fixed-pool
metrics, not directly comparable to official full-corpus BEIR leaderboard scores.
Do not tune on the test split. For training, use `--split train` and a separate
output directory and cache.
