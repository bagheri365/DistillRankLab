"""Evaluate frozen cross-encoder reranking against BM25 on prepared SciFact."""
import argparse
import json
from pathlib import Path

from distillranklab.benchmark import compare_rankers, load_benchmark
from distillranklab.teacher import DEFAULT_MODEL, cross_encoder_scorer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('datasets/scifact_test_bm25'))
    parser.add_argument('--cache', type=Path, default=Path('datasets/scifact_teacher_scores.json'))
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--unjudged', choices=('nonrelevant', 'judged_only'), default='nonrelevant')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()

    groups, manifest = load_benchmark(args.data)
    scorer = cross_encoder_scorer(args.model, device='cpu')
    result = compare_rankers(groups, scorer=scorer, model_id=args.model,
                             cache_path=args.cache, batch_size=args.batch_size,
                             k=args.k, unjudged=args.unjudged)
    result['dataset'] = 'BEIR SciFact'
    result['split'] = manifest['split']
    result['retriever'] = manifest.get('retriever')
    result['top_k_candidates'] = manifest.get('top_k')
    result['archive_md5'] = manifest.get('archive_md5')
    print(json.dumps(result, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
