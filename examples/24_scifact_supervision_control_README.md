# 9A — Controlled distillation supervision ablation

**Research question:** Holding the frozen MiniLM encoder, feature block, linear
head, optimizer, initialization, candidate pool and training split fixed, does
cross-encoder teacher supervision add value beyond hard relevance labels?

Run from the repository root after `pip install -e .` and `pytest`:

```bash
python examples/24_scifact_supervision_control.py \
  --train-data datasets/scifact_full_train_bm25 \
  --teacher-cache datasets/scifact_full_train_teacher_scores.json \
  --embedding-cache datasets/scifact_8b_embeddings.sqlite \
  --output datasets/scifact_9a_supervision_control.json \
  --replicates 10000
```

Two frozen-encoder feature blocks are tested: semantic-only (2 active head
coordinates) and combined (10). For each, `teacher_only` uses soft teacher
scores (hard weight 0), `relevance_only` uses judged-positive listwise targets
(hard weight 1), and `hybrid` uses both (hard weight 0.5). The semantic-only
block also includes a zero-fit raw cosine baseline. Every fitted head has
identical initialization seed 42, 120 optimization steps, temperature 2,
learning rate 0.05 and the same candidate pool.

Hard targets are constructed only for queries with retrieved judged positives.
Unjudged candidates receive zero hard gain **as an assumption**, not as a
verified negative label. Teacher-only still depends on a pretrained frozen
encoder, so this ablation isolates **supervision within the fixed architecture**,
not all pretrained knowledge. Hard-only requires teacher scores to be present
in the shared API but their loss coefficient is zero. No new teacher or
embedding inference is allowed. The official test set is never used.

All reported paired CIs and p-values are **exploratory and unadjusted**, on the
same validation split used in milestones 7–8. The report is not a confirmatory
claim of distillation benefit; model selection on this split would further
increase selection bias. A genuinely new benchmark or untouched queries and
CPU end-to-end timing are needed before final conclusions.
