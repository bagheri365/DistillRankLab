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
    parser.add_argument('--per-query-output', type=Path, default=None,
                        help='Write per-query improvements/regressions to a separate JSON file')
    args = parser.parse_args()

    groups, manifest = load_benchmark(args.data)
    scorer = cross_encoder_scorer(args.model, device='cpu')
    result = compare_rankers(groups, scorer=scorer, model_id=args.model,
                             cache_path=args.cache, batch_size=args.batch_size,
                             k=args.k, unjudged=args.unjudged,
                             per_query=args.per_query_output is not None)
    result['dataset'] = 'BEIR SciFact'
    result['split'] = manifest['split']
    result['retriever'] = manifest.get('retriever')
    result['top_k_candidates'] = manifest.get('top_k')
    result['archive_md5'] = manifest.get('archive_md5')
    if args.per_query_output is not None:
        diagnostics = result.pop('per_query')
        args.per_query_output.parent.mkdir(parents=True, exist_ok=True)
        args.per_query_output.write_text(json.dumps(diagnostics, indent=2) + '\n', encoding='utf-8')
        result['per_query_output'] = str(args.per_query_output)
        result['ndcg_improved_queries'] = diagnostics['ndcg_improved_queries']
        result['ndcg_regressed_queries'] = diagnostics['ndcg_regressed_queries']
        result['ndcg_unchanged_queries'] = diagnostics['ndcg_unchanged_queries']
    print(json.dumps(result, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
