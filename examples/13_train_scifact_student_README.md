# Milestone 7A — CPU-only real-data student

This experiment trains a small four-feature linear ranker on **official SciFact
train queries**. It does not train or fine-tune the cross-encoder. It supports
`soft` (listwise teacher distribution), `hard` (normalized judged-positive
gains), and `hybrid` (weighted sum) objectives. Soft training includes all
candidates; hard training assumes unjudged candidates have zero relevance,
which is a limitation, **not** an assertion that they are truly negative.
Queries with no retrieved judged positive are omitted from the hard loss,
but remain in the soft loss.

The feature vector uses lexical token overlap, query token recall, passage
length, and the first-stage BM25 rank prior. It has no neural encoder; it is a
transparent, CPU-friendly **proof of concept**, not yet a deployable neural
retrieval student. The model uses no test labels or test teacher scores during
fitting. Train and test splits must be official, disjoint SciFact splits.

First prepare the full training split (recommended to match test top-100):

```bash
python examples/09_prepare_scifact.py --split train --top-k 100 --output datasets/scifact_full_train_bm25
```

Start with a quick **hard-label** run without teacher inference:

```bash
python examples/13_train_scifact_student.py --train-data datasets/scifact_train_bm25 --test-data datasets/scifact_test_bm25 --mode hard --output datasets/scifact_hard_student.json
```

Then train on frozen teacher scores (CPU scoring of the full train pool may
take a long time; cache is reused on subsequent runs):

```bash
python examples/13_train_scifact_student.py --train-data datasets/scifact_full_train_bm25 --test-data datasets/scifact_full_test_bm25 --mode soft --train-cache datasets/scifact_full_train_teacher_scores.json --output datasets/scifact_soft_student.json
python examples/13_train_scifact_student.py --train-data datasets/scifact_full_train_bm25 --test-data datasets/scifact_full_test_bm25 --mode hybrid --train-cache datasets/scifact_full_train_teacher_scores.json --output datasets/scifact_hybrid_student.json
```

For a fair comparison, run hard, soft and hybrid on the **same training pool**,
with identical optimizer settings. These are fixed-pool MRR/nDCG scores with
unjudged candidates treated as nonrelevant; they are not official BEIR
full-corpus metrics. The teacher's full-test results from Milestone 6F are
comparable only when the test pool and evaluation protocol match exactly.

**Research caution:** The test set is for final evaluation, not tuning. To
choose temperature, mixture weight or steps, introduce a validation split
from the official train queries before interpreting final test results.
