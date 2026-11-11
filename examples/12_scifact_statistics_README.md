# Milestone 6G — paired uncertainty for SciFact ranking

Run on the saved per-query results from Milestone 6F. **No model inference or
additional downloads are needed.** Requires only the project's NumPy dependency.

```bash
python examples/12_scifact_statistics.py \
  --per-query datasets/scifact_full_per_query.json \
  --output datasets/scifact_full_statistics.json \
  --replicates 10000 --seed 42
```

For each of MRR@10 and nDCG@10, the output includes the observed mean
teacher-minus-BM25 difference, a query-paired percentile bootstrap 95% interval,
and a two-sided paired sign-flip randomization p-value (Monte Carlo with +1
correction). Bootstrap queries are sampled *as pairs*, preserving within-query
correlation. Queries with no retrieved judged positives remain in the sample
with zero credit for both rankers.

The sign-flip test assumes exchangeability of the two ranker labels under the
null; the percentile interval is not an exact finite-sample interval. This is
an exploratory comparison of two metrics, **not** a multiple-testing-adjusted
confirmatory study. Incomplete SciFact judgments and the fixed BM25 candidate
pool limit generalizability; do not compare directly to full-corpus BEIR scores.
The output is reproducible for the same NumPy version, seed, and input JSON.
