"""Inspect the bundled illustrative TSV data; no network or model required."""
from pathlib import Path

from distillranklab.real_data import judgment_coverage, load_groups, split_groups


def main() -> None:
    base = Path(__file__).resolve().parents[1] / 'tests' / 'fixtures' / 'tiny_ranking'
    groups = load_groups(*(base / name for name in (
        'queries.tsv', 'passages.tsv', 'candidates.tsv', 'qrels.tsv')))
    train, test = split_groups(groups, test_fraction=0.4, seed=42)
    print('Illustrative fixture (NOT an MS MARCO sample)')
    print(f'Queries: {len(groups)}; train: {len(train)}; test: {len(test)}')
    print('Judgment coverage:', judgment_coverage(groups))
    first = groups[0]
    print(f'Example query {first.query_id}: {first.query}')
    for candidate in first.candidates:
        label = 'unjudged' if candidate.relevance is None else str(candidate.relevance)
        print(f'  {candidate.passage_id}: relevance={label} | {candidate.passage}')
    print('For real data, provide your own four TSV files to load_groups().')


if __name__ == '__main__':
    main()
