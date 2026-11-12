# Milestone 7D — nonlinear student capacity ablation

This is a **train-only, exploratory** experiment. It does not load the official
SciFact test set, and the previously reused validation split is **not an
untouched holdout**. All three models use the same eight rich lexical features,
listwise hybrid objective (50% hard, 50% soft), temperature 2, learning rate
0.05, 120 full-batch steps, seed 42, and the same 647/162 query-disjoint split.
The only controlled variable is architecture/hidden width:

| Architecture | Trainable parameters | Details |
|---|---:|---|
| Linear | 8 | No intercept, existing 7C implementation |
| MLP 8 | 81 | `tanh(X W1 + b1) W2 + b2` |
| MLP 16 | 161 | Same architecture, wider hidden layer |

These counts include all biases. The MLP uses only NumPy, no neural frameworks,
and query-averaged full-batch listwise gradients. The hard objective assumes
unjudged documents are nonrelevant; this assumption is not a ground-truth label.
The input includes the original BM25 candidate-order prior. Parameter counts
are not comparable to the frozen cross-encoder's parameter count.

## Run

```bash
pytest
python examples/16_scifact_nonlinear_ablation.py \
  --train-data datasets/scifact_full_train_bm25 \
  --train-cache datasets/scifact_full_train_teacher_scores.json \
  --output datasets/scifact_nonlinear_ablation.json \
  --student-output datasets/scifact_nonlinear_student.json
```

The teacher cache must be complete. The script refuses to infer missing teacher
scores unless explicitly passed `--allow-inference`, which can take substantial
CPU time. Training and validation runtimes are measured per configuration, not
across data loading, feature caching, or full-data refitting. The winning model
is refit on all 809 official training queries and saved in JSON format. Models
are selected by validation all-query nDCG@10; ties go to the earlier config.
No official test split is accessed.

## Reporting

Copy actual measured scores, parameter counts, runtime, commit SHA, environment,
and limitations into `docs/experiment_registry.md`. A tiny improvement on this
reused validation split does not establish a statistically significant gain.
Avoid further test-set tuning; a fresh holdout or nested evaluation is needed
for credible generalization claims. Results may be affected by the small
feature representation, candidate-order prior, teacher calibration, and
unjudged-as-negative assumptions.
