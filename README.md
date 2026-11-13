# DistillRankLab

**Does learning to imitate a stronger ranking teacher actually improve relevance ranking?**

DistillRankLab is a controlled research project investigating knowledge distillation for neural search reranking. It separates **teacher imitation**, **relevance ranking**, and **lexical–semantic signal combination** so that an improvement in one is not mistaken for an improvement in another.

The project rule is simple:

> Hold the comparison conditions fixed, measure ranking quality on the same queries, and preserve negative results alongside positive ones.

## At a Glance

- **Research question:** when does cross-encoder teacher supervision improve a smaller student's relevance ranking beyond ordinary relevance supervision?
- **Current research milestone:** **10B — frozen-shortlist reranking robustness audit** (not a release tag; commit [`6e472df`](https://github.com/bagheri365/DistillRankLab/commit/6e472df))
- **Completed empirical benchmark:** SciFact, using BM25 top-100 candidate pools (the original project plan also discussed MS MARCO)
- **Controlled training split:** 647 fit queries and 162 query-disjoint validation queries
- **Primary distillation finding:** students learn teacher preferences, but no reliable incremental validation ranking gain from distillation has been established
- **Exploratory ranking finding:** a 75% semantic / 25% BM25 rank blend reaches **0.71990 nDCG@10** versus **0.68836** for BM25 and **0.70291** for cosine alone
- **Critical caveat:** validation has been examined repeatedly, and the official SciFact test set was examined earlier; the final comparisons are **not pristine confirmatory results**
- **Software status:** **384 tests passed** on the development Mac at Milestone 10B

## Why This Project Exists

A strong teacher can supply useful soft targets to a smaller ranking model. But three claims are often conflated:

1. A student **learns to imitate the teacher**.
2. A student **ranks judged relevant passages better**.
3. Distillation **causes a ranking improvement beyond an otherwise comparable student**.

These are distinct empirical questions. DistillRankLab tests them separately, using matched student configurations, explicit supervision controls, query-level ranking metrics, and paired comparisons.

> **Central question:** Does better teacher imitation translate into better held-out relevance ranking?

## Experimental Setup

| Component | Configuration |
|---|---|
| Benchmark | SciFact scientific document retrieval |
| Candidate retrieval | BM25 top-100 passages per query |
| Teacher | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Frozen semantic encoder | `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) |
| Students | Linear and nonlinear ranking heads over lexical and/or semantic features |
| Supervision | Teacher-only soft targets, relevance-only hard targets, hybrid objectives, and confidence-weighted variants |
| Fit / validation | 647 / 162 query-disjoint queries, fixed split |
| Primary ranking metrics | nDCG@10 and MRR@10, including all queries |
| Statistical diagnostics | Query-paired bootstrap intervals and sign-flip tests |

The full validation candidate pool contains **16,200 passages**, of which **170 are judged** and **16,030 are unjudged**. Twelve of the 162 queries have no judged relevant passage retrieved in their top-100 candidate pool. These facts materially limit interpretation of ranking metrics.

## Key Findings

### 1. Teacher imitation works; a distillation-specific relevance gain is not established

Controlled comparisons held the encoder, student head, data, training budget, and seed fixed while varying supervision.

| Combined-feature student objective | Validation nDCG@10 |
|---|---:|
| Relevance-only | 0.69720 |
| Teacher-only | 0.69807 |
| Hybrid | 0.69874 |
| Frozen cosine, without a trained head | **0.70291** |

The hybrid objective exceeds relevance-only supervision by approximately **+0.00154 nDCG@10**, but the paired interval includes zero and the sign-flip result does not establish a reliable difference (`p = 0.17848`). Teacher-only supervision likewise shows no reliable gain over relevance-only (`p = 0.56754`).

Training diagnostics confirm that teacher-supervised students **reduce teacher-imitation KL loss**. The absence of a demonstrated relevance-ranking improvement therefore cannot simply be described as training failure.

In the semantic-only comparison, teacher-only, hybrid, and relevance-only students yielded the **same validation ranking metrics as raw cosine**. Equal aggregate metrics do not establish identical full rankings or learned weights.

### 2. A better teacher does not automatically make a better distilled ranker

An earlier official SciFact test comparison found:

| Ranker | nDCG@10 | MRR@10 |
|---|---:|---:|
| BM25 | 0.66387 | 0.62762 |
| Cross-encoder teacher | 0.68090 | 0.65092 |

The teacher's observed nDCG@10 difference was approximately **+0.0170**, with a paired interval spanning zero and a sign-flip `p ≈ 0.267`.

The official test set was examined during the project and **must not be presented as an untouched final confirmation set**.

### 3. Teacher–relevance alignment is query-dependent

A fit-split diagnostic showed that the teacher can rescue judged relevant passages originally ranked below the BM25 top positions, but can also disturb strong BM25 rankings. The analysis is **descriptive**: qrel-derived categories were not used as inference-time routing signals, and unjudged passages are not verified negatives.

Confidence-weighted teacher supervision did not improve the held-out combined-feature student's ranking over uniform teacher supervision. A single shuffled-confidence control also did not produce a distinct aggregate ranking outcome.

### 4. Lexical and semantic signals are complementary in exploratory reranking

A separate, **non-distillation** experiment combined normalized BM25 ranks with frozen semantic cosine ranks.

| Reranking strategy | Validation nDCG@10 | Validation MRR@10 |
|---|---:|---:|
| BM25 | 0.68836 | 0.65226 |
| Frozen cosine | 0.70291 | 0.67072 |
| 50% semantic rank blend | 0.71801 | 0.68545 |
| **75% semantic rank blend** | **0.71990** | **0.69266** |

The 75% blend's observed nDCG@10 gain is **+0.03154 versus BM25** and **+0.01699 versus cosine**. The paired confidence intervals include zero (`p ≈ 0.132` and `p ≈ 0.164`, respectively), and multiple exploratory comparisons were made on reused validation data.

**These gains are not evidence of a benefit from knowledge distillation:** the blending rules use no teacher targets or trained distilled student.

### 5. Robustness diagnostics favor blending, with tradeoffs

Milestone 10B froze four strategies—BM25, cosine, 50% blend, and 75% blend—and audited their behavior without searching for new blending weights.

| Strategy vs BM25 | Improved queries | Regressed queries | Unchanged |
|---|---:|---:|---:|
| Cosine | 38 | 33 | 91 |
| 50% blend | 39 | **15** | 108 |
| 75% blend | 39 | 19 | 104 |

Both blends retained a positive mean nDCG@10 difference versus BM25 when any **single** validation query was removed. This excludes a one-query explanation, **not** broader sampling uncertainty or selection bias.

Candidate-pool sensitivity also matters:

| BM25 pool size | 75% blend nDCG@10 | Queries with retrieved judged positive |
|---|---:|---:|
| Top 20 | 0.73511 | 141 / 162 |
| Top 50 | 0.72696 | 145 / 162 |
| Top 100 | 0.71990 | 150 / 162 |

Smaller pools can improve measured top-10 ranking while **losing positive-retrieval coverage**. Pool truncation changes both candidate availability and reranking opportunity; it is not a pure ranking-quality intervention.

## Research Evolution

> fundamentals → SciFact benchmark → teacher/student comparisons → semantic features → attribution controls → diagnostics → exploratory hybrid reranking → robustness audit

| Stage | Research question | Main outcome |
|---|---|---|
| Foundations | Can distillation objectives and students be tested reproducibly? | Temperature/KL objectives, synthetic ablations, ranking tests |
| SciFact baseline | How do BM25 and a pretrained teacher compare? | Teacher numerically ahead; uncertainty remains |
| Student development | Do lexical and semantic student heads improve ranking? | Frozen cosine proved a strong baseline |
| 9A — Supervision control | Is the gain attributable to teacher supervision? | No reliable distillation-specific relevance gain |
| 9B — Learning diagnostics | Did teacher-supervised training actually learn? | Teacher imitation improved without corresponding ranking evidence |
| 9C — Teacher alignment | Where does teacher ranking help or hurt? | Query-dependent rescue and regression patterns |
| 9D — Confidence supervision | Does weighting teacher signals help? | No demonstrated improvement over uniform supervision |
| 10A — Rank blending | Can lexical and semantic ranks complement each other? | Promising exploratory 75% semantic blend |
| 10B — Robustness | Are results driven by one query or one pool size? | Positive leave-one-out means; candidate-coverage tradeoffs |

## Negative and Inconclusive Results

DistillRankLab intentionally retains outcomes that do **not** support the original hypothesis:

- Teacher imitation improved during training, but its incremental relevance-ranking value was not established.
- Hybrid and teacher-only supervision did not reliably beat relevance-only supervision under matched conditions.
- Confidence-aware distillation did not improve validation ranking over uniform supervision.
- A simple BM25 top-1 protection heuristic reduced aggregate performance at higher semantic blending weights.
- The best observed blend was not statistically confirmed as superior to either baseline.

These findings constrain the claims; they are not discarded experiments.

## Evaluation Discipline and Limitations

- **Reused validation:** the same 162 queries were inspected across many milestones; reported comparisons are exploratory.
- **Previously inspected official test:** an earlier official SciFact test evaluation exists; it is not a pristine final holdout.
- **Sparse judgments:** unjudged candidates are counted nonrelevant in the reported ranking metrics, but should not be interpreted as verified negatives.
- **Limited scope:** one benchmark and a limited collection of teachers, frozen encoders, student heads, and supervision schemes.
- **Multiple comparisons:** exploratory paired p-values are not adjusted for the complete research search process.
- **No efficiency claim:** cached embeddings avoid recomputation, and cache timing is not end-to-end CPU inference latency.
- **No distillation attribution for blending:** the strongest observed hybrid ranker is a fixed combination of existing signals.

A genuinely unseen benchmark would be required for independent confirmation and stronger generalization claims.

## Reproducibility

The repository contains experiment scripts, tests, research notes, and deterministic controls. The completed empirical work focuses on SciFact; MS MARCO was part of the initial project plan, not the source of the results reported here. Some experiments require SciFact datasets, cached teacher scores, and/or frozen encoder embeddings prepared by earlier milestones. These prerequisites are **not** installed or generated by `pip install -e .`.

The 10A/10B validation pool is identified by SHA-256:

```text
efca1a9fa6463a831749c747b9a3d1543f18161c474aadec63750949aa6cc440
```

### Environment and tests

The development environment used **Python 3.10.15** on an Intel Mac. The package declares version `0.1.0` in `pyproject.toml`; Milestone 10B is not asserted to be a published release tag. For package installation and the unit test suite:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

At Milestone 10B, the developer reported **384 passing tests**. Running the research examples additionally requires their dataset/cache prerequisites and, where applicable, optional model libraries; these are not covered by the base package dependency list.

### Reproduce the exploratory rank-blending evaluation

With the prerequisite dataset and embedding cache already prepared:

```bash
python examples/28_scifact_conservative_reranking.py \
  --train-data datasets/scifact_full_train_bm25 \
  --embedding-cache datasets/scifact_8b_embeddings.sqlite \
  --output datasets/scifact_10a_conservative_reranking.json \
  --replicates 10000
```

### Reproduce the robustness audit

```bash
python examples/29_scifact_reranking_robustness.py \
  --train-data datasets/scifact_full_train_bm25 \
  --embedding-cache datasets/scifact_8b_embeddings.sqlite \
  --output datasets/scifact_10b_reranking_robustness.json
```

For preparation steps and earlier controlled experiments, see the individual `examples/*_README.md` files and the research documentation under `docs/`.

## Repository Structure

```text
DistillRankLab/
├── docs/                  # research protocol, experiment registry, milestone notes
├── examples/              # reproducible experiment entry points and instructions
├── src/distillranklab/     # ranking, distillation, evaluation, and audit code
├── tests/                 # automated unit and regression tests
├── datasets/              # local benchmark inputs, caches, and generated reports (if prepared)
├── pyproject.toml         # Python package and test configuration
└── README.md
```

Large datasets, model artifacts, and local caches may need to be generated or downloaded separately; do not assume they are included in a fresh clone.

## What the Experiments Suggest

The evidence supports three practical distinctions:

1. **Imitation is not relevance.** A lower teacher KL loss does not guarantee a better relevance ranking.
2. **Teacher quality is not enough.** The value of teacher supervision depends on alignment with the target relevance judgments and on the student representation.
3. **Complementary signals may matter more than imitation in this setting.** Simple BM25–cosine rank blending showed promising exploratory results without distillation.

These are conclusions about this experimental setting, not universal rules for neural search.

## Future Research

- Confirm the frozen ranking shortlist on a genuinely unseen benchmark with appropriate relevance judgments.
- Replicate the supervision controls across additional datasets, teachers, and student capacities.
- Investigate teacher–relevance disagreement without treating unjudged documents as known negatives.
- Measure actual encoder and reranker latency if deployment efficiency becomes a research objective.

## Current Conclusion

DistillRankLab asks:

> **Does learning to imitate a stronger ranking teacher actually improve relevance ranking?**

The current evidence says:

> **Teacher imitation can succeed without a demonstrated gain in relevance ranking.**

Controlled distillation experiments did not establish an incremental ranking benefit over comparable relevance-only supervision. Separately, lexical–semantic rank blending produced promising—but exploratory—ranking gains. The project therefore contributes a tested experimental framework, a documented negative result, and a clear boundary between **learning the teacher** and **improving retrieval**.
