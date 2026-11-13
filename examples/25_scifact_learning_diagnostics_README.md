# Milestone 9B — Learning diagnostics (train-only, exploratory)

**Question:** Did the small head learn teacher preferences even when the
validation ranking metrics were unchanged? Inspect the training trajectory,
head coefficients, teacher ranking agreement on **fit queries**, and exact
validation candidate-order changes from the common random initialization.

Run after 9A, with the complete existing train BM25, cross-encoder teacher,
and frozen MiniLM embedding caches:

```bash
pytest
python examples/25_scifact_learning_diagnostics.py \
  --train-data datasets/scifact_full_train_bm25 \
  --teacher-cache datasets/scifact_full_train_teacher_scores.json \
  --embedding-cache datasets/scifact_8b_embeddings.sqlite \
  --output datasets/scifact_9b_learning_diagnostics.json
```

The output records 0/1/10/30/60/120-step checkpoints (where available),
soft-target KL, hard-target cross-entropy, active weights, fit-query teacher
rank agreement, and validation rank-order changes relative to the identical
randomly initialized head. Training objectives are computed **for every mode**
for diagnostic comparability; only the configured terms contribute gradients.
The terminal prints a compact summary; JSON contains full trajectories.

**Important:** Fit-query teacher agreement is in-sample and is not evidence
of teacher imitation on held-out queries. Changes in score or rank order do
not imply relevance improvements. Stable ties use candidate-pool order;
Spearman uses these deterministic ranks. The benchmark has sparse judgments,
repeatedly inspected validation, and no official test access. No new model
inference is allowed. A zero metric difference is not proof of equivalence.
