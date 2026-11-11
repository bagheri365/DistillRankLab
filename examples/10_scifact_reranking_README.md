# Milestone 6E: SciFact BM25 versus frozen cross-encoder

Prepare the genuine BEIR SciFact candidate pool first:

```bash
python examples/09_prepare_scifact.py --download --max-queries 5 --top-k 20
```

Then evaluate BM25 order against cross-encoder reranking of **the same pool**:

```bash
python examples/10_scifact_reranking.py --batch-size 8 --output datasets/scifact_5q_results.json
# Repeat to verify cached scores are reused; this is not a new inference benchmark.
python examples/10_scifact_reranking.py --batch-size 8
```

For a larger test set, regenerate the candidate pool in a **new directory**,
using `--max-queries 50 --top-k 100` or the full split (omit `--max-queries`).
Set `--data` to that directory and use a separate cache if desired. CPU inference
can take substantial time. The script lazily imports the optional
`sentence-transformers` dependency only when run.

The baseline reconstructs BM25 *order* from candidates.tsv; it does not
reconstruct numerical BM25 scores. The `nonrelevant` policy retains unjudged
passages with zero gain; `judged_only` removes them before ranking. Both are
assumptions about incomplete judgments, not verified relevance.

**Conditional** MRR@k/nDCG@k averages only queries with a judged positive
in the candidate pool. **All-query** averages include those without any
retrieved judged positive as zero. The coverage field counts how many queries
had at least one judged positive in the pool. These are *fixed-candidate-pool*
results, not full-corpus BEIR leaderboards; candidate recall is limited by BM25.
The timing field excludes model initialization and includes cache I/O. On a
warm cache it is **not** a measurement of neural inference latency.

Do not select hyperparameters using the held-out test queries. Keep the
prepared datasets, score caches, and result JSON files local under `datasets/`.
