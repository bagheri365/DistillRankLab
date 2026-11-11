# Real-data foundation

Run `python examples/06_real_data_foundation.py` after `pip install -e ".[dev]"`.

The bundled TSV files are **illustrative fixtures, not MS MARCO data**.
The loader expects four UTF-8 TSV files without headers:

- `queries.tsv`: `query_id<TAB>query_text`
- `passages.tsv`: `passage_id<TAB>passage_text`
- `candidates.tsv`: `query_id<TAB>passage_id`
- `qrels.tsv`: `query_id<TAB>passage_id<TAB>relevance` (nonnegative integer)

Missing qrels are **unjudged** (`None`), not automatically irrelevant.
A qrel with relevance `0` is explicitly judged nonrelevant. The loader
retains candidate order and validates duplicate IDs and missing references.

Use `split_groups` to make query-disjoint train/test sets. This is only a
foundation for a ranking benchmark: candidate selection, incomplete judgments,
and the provenance of qrels must be documented before reporting metrics.
The loader does not download datasets or train a neural model.
