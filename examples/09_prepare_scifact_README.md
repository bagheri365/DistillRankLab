# Milestone 6D: BEIR SciFact benchmark preparation

Official dataset: https://github.com/beir-cellar/beir/wiki/Datasets-available

```bash
python examples/09_prepare_scifact.py --download --max-queries 20 --top-k 100
python examples/09_prepare_scifact.py --split train --output datasets/scifact_train_bm25 --max-queries 20
```

The downloader checks the published SciFact archive MD5. Outputs contain the four TSV
files accepted by `distillranklab.real_data.load_groups`, plus `manifest.json`.
The train and test splits are BEIR's official splits, **not** a random split.
BM25 candidates are generated from corpus and query text **without qrels**;
judgments are attached afterward. **No positive injection** occurs. Queries
with no retrieved positive must be counted as retrieval failures in an end-to-end
benchmark, not silently discarded. Existing ranking metrics that exclude such
queries describe *conditional reranking* only, and must not be presented as
full-corpus BEIR retrieval metrics.

The default is 20 queries for a quick CPU smoke test; omit `--max-queries`
for the full split. The output is an approximate, BM25-first-stage candidate
pool and is not directly comparable to BEIR full-corpus retrieval leaderboards.
Teacher scoring on 20 × 100 candidates may be slow on CPU; start with
`--max-queries 5 --top-k 20` to test the integration.

The SciFact archive and generated TSV files are local data artifacts and should
not be committed to Git. Consult the dataset's license and citation requirements
before distributing data or publishing experimental results.
