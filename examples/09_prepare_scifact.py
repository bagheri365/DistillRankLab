"""Download and prepare the official BEIR SciFact dataset without model inference."""
import argparse
import json
from pathlib import Path

from distillranklab.scifact import download_scifact, prepare_scifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path('datasets/scifact.zip'))
    parser.add_argument('--output', type=Path, default=Path('datasets/scifact_test_bm25'))
    parser.add_argument('--split', choices=('train', 'test'), default='test')
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--max-queries', type=int, default=None,
                        help='Optional number of queries; omitted means the full official split')
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    if args.download:
        download_scifact(args.archive)
    if not args.archive.exists():
        parser.error('archive missing; pass --download to retrieve and verify it')
    manifest = prepare_scifact(args.archive, args.output, split=args.split,
                               top_k=args.top_k, max_queries=args.max_queries)
    print(json.dumps(manifest, indent=2))
    print('Prepared TSV files:', args.output)


if __name__ == '__main__':
    main()
