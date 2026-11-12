# Milestone 7C — CPU-only feature ablation

This experiment compares **baseline**, **lexical**, and **rich** linear student
feature sets on **the same 647/162 query-disjoint SciFact train/validation
split** as Milestone 7B (SHA-256 seed 42). No official test data is loaded.

## Run

```bash
pytest
python examples/15_scifact_feature_ablation.py \
  --train-data datasets/scifact_full_train_bm25 \
  --train-cache datasets/scifact_full_train_teacher_scores.json \
  --output datasets/scifact_feature_ablation.json \
  --student-output datasets/scifact_feature_student.json
```

The full train teacher cache from 7A is reused. Missing scores raise an error
unless `--allow-inference` is explicitly supplied. No GPU is required.

The script prints all nine validation trials and the selected configuration.
The JSON records query IDs, BM25 validation baseline, feature definitions,
teacher model, split seeds, and selection rationale. It refits the selected
configuration on all 809 official train queries, without looking at test data.

## Feature definitions

- **baseline (4):** log token overlap, unique query-token recall, log passage
  length, and reciprocal BM25 candidate position.
- **lexical (6):** baseline plus saturated query-token frequency and exact
  query-bigram recall.
- **rich (8):** lexical plus character-trigram Jaccard similarity and a
  bounded query/document token-length ratio.

All features are deterministic and label-free. No corpus-wide statistics or
validation-document IDF are fitted. BM25 candidate rank is an input feature;
therefore this is *not* an independent retrieval model or a neural encoder.

## Fixed ablation grid

Three feature sets × hard, soft, and hybrid objectives = nine trials.
All trials use learning rate 0.05, 120 steps, temperature 2.0 and seed 42;
hybrid hard weight is 0.5. Selection uses all-query validation nDCG@10.
The hard objective assumes unjudged documents are nonrelevant. Soft targets
use frozen teacher scores on the official train split only.

## Interpretation

Compare each trial against BM25 **on validation**, not test. The validation
split and objective were already examined in 7B, so 7C is exploratory;
results should not be presented as unbiased confirmatory evidence. Changes
in feature count are confounded with feature content, and optimizer settings
are held fixed rather than retuned for each feature set. Record actual results
in `docs/experiment_registry.md` after running, rather than inventing scores.
