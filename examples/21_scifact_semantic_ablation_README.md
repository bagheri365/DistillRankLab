# Milestone 8C — Semantic feature block ablation and paired uncertainty

**Question:** On the fixed BM25 top-100 SciFact training-validation candidate pool, do the
frozen-encoder features add useful ranking information beyond lexical features?

**Design:** Official SciFact **train** only, query-disjoint SHA-256 seed-42 split
(647 fit / 162 validation). Three 10-coordinate heads share exactly the same
optimizer, seed 42, 120 steps, temperature 2, hybrid hard weight 0.5, and
learning rate 0.05. Lexical-only zeros the two semantic columns; semantic-only
zeros the eight lexical columns; combined uses all ten. Effective parameter
counts are 8, 2, 10 respectively. Zeroed weights cannot affect ranking.

The frozen `sentence-transformers/all-MiniLM-L6-v2` embeddings and frozen
cross-encoder teacher scores are **reused**, not recomputed when caches are
complete. The script also refits the MLP-8 lexical baseline at the same settings.

```bash
pytest
python examples/21_scifact_semantic_ablation.py \
  --train-data datasets/scifact_full_train_bm25 \
  --teacher-cache datasets/scifact_full_train_teacher_scores.json \
  --embedding-cache datasets/scifact_8b_embeddings.sqlite \
  --output datasets/scifact_8c_ablation.json \
  --replicates 10000
```

Only if embeddings are missing, explicitly opt into downloading/encoding with
`--allow-model-download`. No new teacher inference is allowed.

Paired query-level bootstrap percentile 95% confidence intervals and Monte
Carlo sign-flip tests (10,000 replicates, seed 42) compare combined vs BM25,
combined vs MLP-8, semantic-only vs lexical-only, and combined vs each
individual block. The output uses `mean_challenger_minus_baseline`, not the
legacy `mean_teacher_minus_bm25` field. The same query IDs and candidate pools
are required for every comparison.

**Limits:** Repeatedly inspected validation split; multiple comparisons without
multiplicity adjustment; sparse judgments with unjudged treated nonrelevant;
fixed BM25 candidate pool rather than full-corpus retrieval. Confidence
intervals describe query sampling conditional on this one training seed, not
training-seed uncertainty. Do not treat nominal p-values as confirmatory.
Do not select settings using the official test set.

**Record after running:** all three metrics, BM25 and MLP-8 metrics, paired
deltas/CIs/p-values, embedding cache hits and CPU runtimes in
`docs/experiment_registry.md`. Do not replace TBD with invented values.
