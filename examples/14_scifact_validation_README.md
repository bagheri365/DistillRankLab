# Milestone 7B — Train-only validation and documented student ablation

## Research question

Can the four-feature linear SciFact student be improved by choosing its
listwise training objective and optimizer settings **without using the official
SciFact test set for selection**? Compare hard, soft and hybrid supervision
under identical candidate pools, feature extraction and train/validation split.

## Dataset and leakage controls

- Source: official **BEIR SciFact train** queries, exported by
  `examples/09_prepare_scifact.py` with BM25 top-100 candidates. The `test`
  split is never passed to the selection function.
- A deterministic SHA-256 query-ID split (seed 42) reserves 20% of official
  train queries for validation. The remaining 80% fit each candidate model.
  The split is query-disjoint and independent of file ordering.
- All trials use **the same** split, features, candidate order, and seed.
- Select by **validation all-query nDCG@10**, including queries with no
  retrieved judged positive as zeros. Ties within 1e-12 select the first
  configuration in documented grid order.
- After selection, refit the chosen configuration on **all official train
  queries**. The reported validation score belongs to the pre-refit model;
  it is not a score for the refitted model.
- `--test-data` is optional and intended for **one final evaluation only**
  after the grid and selection rule are frozen. Do not modify hyperparameters
  in response to test outcomes. Existing 7A test comparisons were exploratory;
  treat later test comparisons accordingly.

## Model and objectives

The student is a four-feature NumPy linear ranker: token overlap, query token
recall, passage length, and BM25 rank prior. It is **not** a neural encoder.
Hard supervision uses normalized graded judged-positive gains; unjudged
candidates are assumed to have zero gain **for this objective only**. Queries
without retrieved judged positives are excluded from hard-loss training. Soft
supervision uses frozen cross-encoder scores converted to listwise probability
vectors. Hybrid mixes hard and soft losses. Soft targets require all candidate
scores, not only qrels-positive pairs. BM25 candidate selection never injects
judged positives. This is a fixed-pool reranking experiment, **not** an official
full-corpus BEIR score.

## Run: validation-only (recommended first)

Your Milestone 7A run already populated
`datasets/scifact_full_train_teacher_scores.json` for 80,900 pairs.
The script uses **cache-only** teacher scoring by default, so it does not
silently start another hour-long CPU inference run. If cache entries are
missing, rerun with `--allow-inference` to compute them.

```bash
pytest
python examples/14_scifact_validation.py \
  --train-data datasets/scifact_full_train_bm25 \
  --train-cache datasets/scifact_full_train_teacher_scores.json \
  --output datasets/scifact_validation_report.json \
  --student-output datasets/scifact_selected_student.json
```

Default grid (7 trials): learning rate 0.05; 120 gradient steps; hard at
weight 1; soft at temperatures 1 and 2; hybrid at temperatures 1 and 2 with
hard weights 0.25 and 0.5. All use training seed 42. These are **initial
choices**, not claimed optimal. For a broader *precommitted* grid:

```bash
python examples/14_scifact_validation.py \
  --learning-rates 0.025 0.05 \
  --temperatures 1 2 4 \
  --hard-weights 0.25 0.5 0.75 \
  --steps 120 300
```

The broader grid is slower; start with the default seven trials on CPU.
The report contains every configuration and validation result, the query IDs
in each partition, selected configuration, selection metric, seed, candidate
manifest, and baseline validation metrics. Model JSON records its selected
configuration. The JSON files are in `datasets/` and should not be committed.

## Final evaluation (run once after selection)

Freeze the grid first. Then run the **same command** with `--test-data`:

```bash
python examples/14_scifact_validation.py \
  --train-data datasets/scifact_full_train_bm25 \
  --train-cache datasets/scifact_full_train_teacher_scores.json \
  --test-data datasets/scifact_full_test_bm25 \
  --output datasets/scifact_validation_final_report.json \
  --student-output datasets/scifact_selected_student.json
```

The `final_test` field compares the selected student with BM25 on the same
fixed test candidate pool. For frozen-teacher scores, use the separate
Milestone 6F result, ensuring the candidate manifest and metric policy match.

## Interpretation and reporting checklist

Report: dataset and archive checksum; official split; BM25 top-k; query counts;
retrieved-positive and judgment coverage; grid and seeds; validation selection
metric; each trial's validation score; final test metrics; model architecture;
teacher model and score-cache provenance; unjudged-document policy; and CPU
inference/training limitations. Do **not** claim that an exploratory improvement
is statistically established without paired uncertainty analysis. Repeated
trial selection on a single validation split may overfit that split. The
teacher's previous test improvement over BM25 was **not statistically
significant** under the Milestone 6G analysis.

## Future work

Compare multiple validation seeds and richer lexical/semantic student features,
then use a locked test protocol for final claims. The student is intended as a
reproducible distillation baseline, not proof that distillation beats BM25.
