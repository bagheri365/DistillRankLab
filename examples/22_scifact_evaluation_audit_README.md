# Milestone 8D — Fixed-pool evaluation audit

This experiment **replays the 8C training configuration** on the full official
SciFact **train** benchmark, using the deterministic 647-fit/162-validation
split, 100 BM25 candidates per query, frozen MiniLM embeddings, and the
existing teacher cache. It never reads the official test split.

## Run on the CPU-only Intel Mac

```bash
pytest
python examples/22_scifact_evaluation_audit.py \
  --train-data datasets/scifact_full_train_bm25 \
  --teacher-cache datasets/scifact_full_train_teacher_scores.json \
  --embedding-cache datasets/scifact_8b_embeddings.sqlite \
  --output datasets/scifact_8d_audit.json
```

The command **refuses** missing teacher scores or embeddings instead of
triggering inference. It creates a detailed JSON report and prints a compact
summary. The detailed report is ignored under `datasets/` by Git.

## What is audited

1. Every ranker must have exactly the same (query ID, passage ID) score keys;
   missing/extra, nonfinite, and malformed scores fail fast.
2. An order-sensitive SHA256 fingerprint captures query IDs, candidate IDs,
   their order, and judgment status. Query groups are sorted before hashing.
3. Every per-query MRR@10/nDCG@10 mean must equal its corresponding all-query
   aggregate to within `1e-10` for **both** unjudged policies.
4. Per-query delta counts, standard deviation, median, range and quantiles
   explain whether a small mean hides large wins and losses. Wide bootstrap
   intervals may reflect high query-level variance; the audit does not assume
   an explanation before inspecting the data.
5. `nonrelevant` keeps unjudged candidates with zero gain; `judged_only`
   **removes** them before ranking and cutoff. The latter is a sensitivity
   scenario, **not** an unbiased estimate of relevance and not a replacement
   for additional qrels. The two policies answer different questions.

No p-values are recomputed in 8D: the 8C unadjusted p-values are exploratory,
and the repeatedly reused validation split does not establish generalization.
No model is selected from official test data.
