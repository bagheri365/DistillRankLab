"""9C: audit teacher–relevance alignment on SciFact fit queries only.

Requires full TRAIN BM25 candidate pool and complete frozen teacher cache.
Never runs new inference or touches the official test set.
"""
import argparse
import json
from pathlib import Path

from distillranklab.benchmark import load_benchmark
from distillranklab.evaluation_audit import candidate_fingerprint
from distillranklab.real_data import split_groups
from distillranklab.teacher import DEFAULT_MODEL, score_groups
from distillranklab.teacher_alignment import teacher_relevance_alignment

EXPECTED_SHA = 'efca1a9fa6463a831749c747b9a3d1543f18161c474aadec63750949aa6cc440'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-data', type=Path, default=Path('datasets/scifact_full_train_bm25'))
    parser.add_argument('--teacher-cache', type=Path, default=Path('datasets/scifact_full_train_teacher_scores.json'))
    parser.add_argument('--output', type=Path, default=Path('datasets/scifact_9c_teacher_alignment.json'))
    args = parser.parse_args()
    groups, manifest = load_benchmark(args.train_data)
    if (manifest.get('split') != 'train' or manifest.get('max_queries') is not None
            or len(groups) != 809 or any(len(g.candidates) != 100 for g in groups)):
        parser.error('full official SciFact TRAIN BM25 top-100 required')
    fit, val = split_groups(tuple(groups), test_fraction=.2, seed=42)
    if len(fit) != 647 or len(val) != 162 or candidate_fingerprint(val) != EXPECTED_SHA:
        parser.error('validation pool differs from audited 8D fingerprint')

    def forbidden(_):
        raise RuntimeError('incomplete teacher cache; inference prohibited')

    teacher = score_groups(fit, scorer=forbidden, model_id=DEFAULT_MODEL,
                           cache_path=args.teacher_cache, batch_size=16)
    audit = teacher_relevance_alignment(fit, teacher)
    report = {'experiment': '9C teacher–relevance alignment audit', 'exploratory': True,
              'official_test_used': False, 'teacher_model': DEFAULT_MODEL,
              'fit_queries': len(fit), 'validation_queries_not_evaluated': len(val),
              'validation_pool_sha256': EXPECTED_SHA, 'audit': audit}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'report': str(args.output), 'experiment': report['experiment'],
                      'official_test_used': False, 'fit_queries': len(fit),
                      'validation_queries_not_evaluated': len(val),
                      'bm25_ndcg_at_10': audit['bm25']['ndcg_at_k_all_queries'],
                      'teacher_ndcg_at_10': audit['teacher']['ndcg_at_k_all_queries'],
                      'comparison': audit['comparison'],
                      'teacher_top1_judgments': audit['teacher_top1_judgments'],
                      'teacher_bm25_top1_disagreement': audit['teacher_bm25_top1_disagreement'],
                      'positive_rank_summary': audit['positive_rank_summary'],
                      'by_bm25_positive_rank': audit['by_bm25_positive_rank'],
                      'judged_zero_margin_available': audit['judged_zero_margin_available'],
                      'top_gains': audit['top_gains'], 'top_losses': audit['top_losses']}, indent=2))


if __name__ == '__main__':
    main()
