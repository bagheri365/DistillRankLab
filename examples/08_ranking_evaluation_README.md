# Ranking evaluation (Milestone 6C)

Run the offline fixture evaluation:

```bash
python examples/08_ranking_evaluation.py
python examples/08_ranking_evaluation.py --unjudged judged_only
```

Run the real frozen teacher (requires the optional `sentence-transformers` stack):

```bash
python examples/08_ranking_evaluation.py --real --cache teacher_scores_real.json
```

The example uses a deterministic, query-disjoint holdout from **illustrative data**, not MS MARCO. Scores are cached with the same model/text-aware cache used in Milestone 6B. Do not publish these metrics as real benchmark results.

## Definitions and limitations

- **MRR@10**: reciprocal rank of the first *judged relevant* candidate (relevance > 0) in the top ten, averaged over eligible queries.
- **nDCG@10**: discounted gain using `2**relevance - 1`, normalized by the ideal ranking of judged positive candidates in the same candidate pool; macro-averaged over eligible queries.
- **`nonrelevant`** (default): unjudged passages stay in the ranking and contribute zero gain. This is a conventional evaluation assumption, **not** a statement that they are truly irrelevant.
- **`judged_only`**: unjudged passages are removed *before* ranking and applying the cutoff. This is a different evaluation protocol and can inflate scores; report the policy alongside all metrics.
- Queries with no judged relevant candidate in the candidate pool are **excluded** from both means, with counts reported. A zero result with zero eligible queries is **not** evidence of poor ranking performance.
- Tied scores are ordered by passage ID, not by file order.
- Results apply only to the **provided candidate pool**, not full-corpus retrieval. Candidate generation and judgment incompleteness can bias results. Report query counts, candidate counts, and judgment coverage.

Next milestone: load a real benchmark with documented candidate generation, train/test split provenance, and relevance judgments. Student comparisons should use the exact same held-out groups and metric policy.
