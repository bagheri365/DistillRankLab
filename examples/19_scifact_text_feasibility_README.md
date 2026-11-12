# Milestone 8A — Frozen text encoder feasibility (CPU)

**Question:** Can pretrained semantic query/passage embeddings supply information
beyond the eight handcrafted lexical features? This is a feasibility probe,
**not** a benchmark result or an encoder fine-tuning experiment.

- Frozen `sentence-transformers/all-MiniLM-L6-v2` on CPU, optional dependency.
- Features: eight rich lexical features plus embedding cosine and maximum
  normalized coordinate product. A **10-parameter trainable linear head** is
  fitted with the existing hybrid listwise teacher/hard-label objective.
- Official SciFact **train only**; stable 647/162 query split, then first 12 fit
  queries, 4 validation queries and 20 BM25-ordered candidates per query by
  default. No positive injection, no test-set access.
- Teacher scores must already be cached. Encoder weights may be downloaded
  only after explicit `--allow-model-download`. Text embeddings are cached
  with the model ID and hashed text keys; do not commit caches.
- Validation subset and candidate truncation differ from the earlier top-100
  evaluation. **Never compare the resulting metric directly with 7E numbers.**
  Many truncated queries may lack retrieved positives. The split has also been
  used repeatedly, so results are exploratory.

Install the already-tested CPU environment (Python 3.10, NumPy 1.26):

```bash
python -m pip install 'sentence-transformers==3.0.1'
```

First run (downloads pretrained encoder weights if absent):

```bash
python examples/19_scifact_text_feasibility.py --allow-model-download
```

Subsequent runs can reuse the embedding and teacher caches without model
loading or downloads:

```bash
python examples/19_scifact_text_feasibility.py
```

Outputs: `datasets/scifact_8a_feasibility.json`,
`datasets/scifact_8a_head.json`, `datasets/scifact_8a_embeddings.json`.
Report includes subset size, BM25 on **the same truncated candidate pool**,
student metrics, embedding time and head training time. Cache-warm embedding
wall time measures cache I/O, **not encoder inference latency**. The head is
trained only on the fit subset; validation is used for evaluation, not tuning.

**Next milestone:** assess encoder throughput/memory and expand query coverage
only after documenting feasibility and fixing an evaluation protocol. Do not
use the repeatedly examined SciFact test set to select architecture.
