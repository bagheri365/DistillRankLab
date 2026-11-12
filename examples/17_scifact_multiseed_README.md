# 7E — Multi-seed robustness (train-only)

Repeat the **same fixed rich-feature hybrid objective** for linear, 8-hidden-unit
MLP and 16-hidden-unit MLP. The only change within each architecture is the
random training seed. No test split is loaded or used for model selection.

```bash
python examples/17_scifact_multiseed.py \
  --train-data datasets/scifact_full_train_bm25 \
  --train-cache datasets/scifact_full_train_teacher_scores.json \
  --output datasets/scifact_multiseed.json
```

Default training seeds: `11 23 42 67 101`; split seed: `42` (647 fit / 162
validation queries); steps: `120`; 10,000 paired-query bootstrap/sign-flip
replicates. The full experiment trains 15 models and may take **several minutes
or longer** on an Intel CPU. For a quick smoke test, use `--seeds 11 42 --steps 3
--replicates 100` but **do not interpret its metrics as the full experiment**.
The script requires the existing training teacher cache by default; it will not
silently start CPU cross-encoder inference. Use `--allow-inference` only if
missing training scores need to be computed.

Output contains all per-seed metrics and timings, mean and **sample standard
deviation across training seeds**, and paired query bootstrap confidence
intervals/sign-flip p-values for each MLP relative to the linear student.
Paired query statistics are computed on **per-query metric values averaged across
training seeds**, not on averaged model scores/rankings. They condition on the
specific seeds; they do not quantify uncertainty across seeds. Conversely,
seed standard deviations describe initialization sensitivity but do not quantify
query sampling uncertainty. No correction is made for two architectures and
two metrics; significance claims are inappropriate on this repeatedly used
validation split. The code intentionally does not select a model or touch the
official test split.

Update `docs/experiment_registry.md` with the actual measured values, seed
variability, paired uncertainty, runtimes, commit SHA and environment.
