# 8B — Scaled frozen semantic features (exploratory)

**Question:** On the same SciFact BM25 top-100 pools and the established
647/162 official-train split, does a frozen sentence-embedding feature head
improve validation ranking over the rich linear and MLP-8 baselines?

All models use hybrid listwise training, temperature 2, hard weight 0.5,
learning rate 0.05, 120 steps, and seed 42. The semantic model uses 8 rich
lexical features plus cosine similarity and maximum normalized coordinate
product from frozen `sentence-transformers/all-MiniLM-L6-v2`. Only its
10-parameter head is trained. The teacher is the **already cached**
`cross-encoder/ms-marco-MiniLM-L-6-v2`. No official test split is loaded.

## Run (Intel Mac CPU)

```bash
pytest
python examples/20_scifact_scaled_text.py --allow-model-download
```

The first run may require encoding tens of thousands of unique passages and
can take substantial CPU time and disk space. Progress is printed every ~500
new texts; SQLite checkpoints each batch, so rerun the same command after an
interruption. It will reuse previously completed embeddings. The default
`datasets/scifact_8b_embeddings.sqlite` is ignored by Git through `datasets/`.

After the cache is complete, the command works without `--allow-model-download`:

```bash
python examples/20_scifact_scaled_text.py
```

The run requires `datasets/scifact_full_train_bm25` and
`datasets/scifact_full_train_teacher_scores.json` from earlier milestones.
An incomplete teacher cache **raises** instead of performing teacher inference.
The output is `datasets/scifact_8b_comparison.json` and
`datasets/scifact_8b_head.json`. All three models are evaluated on the
**same 162 queries × 100 BM25 candidates**. Runtimes for embedding retrieval,
feature construction, head fitting, and baseline fitting are separate; cached
runs do not represent cold-start latency. The embedding cache stores float32
vectors and is bound to the specified encoder ID; changing model weights
without changing its ID is not detected. Use a new cache path when changing
weights. The SQLite cache is for a single writer, not concurrent processes.

The previously examined validation set is **exploratory**, not a new holdout.
The top-100 fixed-pool metrics are not full-corpus retrieval metrics. Unjudged
candidates are treated as nonrelevant, an assumption rather than ground truth.
Do not select models using the official test split or claim statistical
significance from this comparison. Record negative results and real timings.
