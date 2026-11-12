# 7F — Reproducible SciFact research summary

No model training, teacher inference, or SciFact test-set evaluation is required.
This step consumes the **full** 7E experiment report already generated on your Mac.

```bash
python examples/18_scifact_research_summary.py \
  --input datasets/scifact_multiseed.json \
  --markdown datasets/scifact_research_summary.md \
  --json-output datasets/scifact_research_summary.json
```

The program checks the predeclared five seeds, 120 steps, 647/162 split,
15 trials, parameter counts, query IDs, and summary/paired-delta consistency.
It rejects the short 7E smoke-test report. Output is deterministic for a
fixed input; timing is copied from the measured experiment, not remeasured.

The Markdown table reports model metrics, seed variation, runtime and
paired query uncertainty. Both output files belong in ignored `datasets/`.
The research registry records actual results independently of this generator.
Validation was reused across several milestones, so no confirmatory claims
should be made from these p-values. This is **not** a full-corpus BEIR score.
