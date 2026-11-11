"""Prepare BEIR SciFact for judgment-aware reranking experiments.

Candidate retrieval never reads qrels; judgments are joined only at export.
"""

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import urllib.request
import zipfile

from .real_data import load_groups

URL = 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip'
MD5 = '5f7d1de60b170fc8027bb7898e2efca1'  # BEIR published archive checksum
TOKEN = re.compile(r'(?u)\b\w+\b')


def download_scifact(archive: str | Path) -> Path:
    """Download the published archive, checking its checksum even if cached."""
    archive = Path(archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        temporary = archive.with_suffix(archive.suffix + '.part')
        try:
            urllib.request.urlretrieve(URL, temporary)
            if _md5(temporary) != MD5:
                raise ValueError('SciFact archive MD5 mismatch')
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    if _md5(archive) != MD5:
        raise ValueError('SciFact archive MD5 mismatch')
    return archive


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def retrieve_bm25(corpus: dict[str, str], queries: dict[str, str], *, top_k: int = 100) -> dict[str, tuple[str, ...]]:
    """Deterministic BM25 candidate generation with stable document-ID ties."""
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError('top_k must be a positive integer')
    if not corpus or not queries:
        raise ValueError('corpus and queries must be nonempty')
    doc_ids = sorted(corpus)
    counts = {pid: Counter(_tokens(corpus[pid])) for pid in doc_ids}
    lengths = {pid: sum(counts[pid].values()) for pid in doc_ids}
    avg_length = sum(lengths.values()) / len(doc_ids)
    df = Counter(term for pid in doc_ids for term in counts[pid])
    result = {}
    for qid, query in sorted(queries.items()):
        terms = set(_tokens(query))
        scores = {}
        for pid in doc_ids:
            score = 0.0
            for term in terms:
                tf = counts[pid][term]
                if tf:
                    idf = math.log(1 + (len(doc_ids) - df[term] + 0.5) / (df[term] + 0.5))
                    score += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * lengths[pid] / max(avg_length, 1)))
            scores[pid] = score
        result[qid] = tuple(sorted(doc_ids, key=lambda pid: (-scores[pid], pid))[:top_k])
    return result


def _jsonl(stream):
    records = {}
    for raw in stream:
        if not raw.strip():
            continue
        item = json.loads(raw)
        key = str(item['_id'])
        if key in records:
            raise ValueError('duplicate JSONL ID')
        records[key] = item
    return records


def _read_archive(archive: Path, split: str):
    if split not in ('train', 'test'):
        raise ValueError('split must be train or test')
    with zipfile.ZipFile(archive) as zf:
        def locate(suffix):
            matches = [name for name in zf.namelist() if name.endswith('/' + suffix) or name == suffix]
            if len(matches) != 1:
                raise ValueError(f'expected one {suffix} in archive')
            return matches[0]
        corpus = _jsonl((line.decode('utf-8') for line in zf.open(locate('corpus.jsonl'))))
        queries = _jsonl((line.decode('utf-8') for line in zf.open(locate('queries.jsonl'))))
        with zf.open(locate(f'qrels/{split}.tsv')) as handle:
            rows = list(csv.reader((line.decode('utf-8') for line in handle), delimiter='\t'))
    if not rows or rows[0] != ['query-id', 'corpus-id', 'score']:
        raise ValueError('unexpected BEIR qrels header')
    qrels = {}
    for row in rows[1:]:
        if len(row) != 3:
            raise ValueError('invalid qrels row')
        qid, pid, relevance = row
        if qid not in queries or pid not in corpus or (qid, pid) in qrels:
            raise ValueError('invalid qrels reference or duplicate')
        value = int(relevance)
        if value < 0:
            raise ValueError('negative relevance')
        qrels[qid, pid] = value
    return corpus, queries, qrels


def prepare_scifact(archive: str | Path, output: str | Path, *, split: str = 'test',
                    top_k: int = 100, max_queries: int | None = None) -> dict:
    """Export a BM25 candidate pool and qrels into the four project TSV files.

    max_queries selects lexicographically first official split IDs, never random
    cross-split mixing. No positive-document injection is performed.
    """
    if max_queries is not None and (not isinstance(max_queries, int) or isinstance(max_queries, bool) or max_queries < 1):
        raise ValueError('max_queries must be positive or None')
    archive = Path(archive)
    corpus, all_queries, qrels = _read_archive(archive, split)
    query_ids = sorted({qid for qid, _ in qrels})
    if max_queries is not None:
        query_ids = query_ids[:max_queries]
    if not query_ids:
        raise ValueError('no judged queries in split')
    texts = {pid: ' '.join(part for part in (str(item.get('title') or ''), str(item.get('text') or '')) if part).strip()
             for pid, item in corpus.items()}
    query_texts = {qid: str(all_queries[qid]['text']) for qid in query_ids}
    candidates = retrieve_bm25(texts, query_texts, top_k=top_k)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    def write(name, rows):
        with (output / name).open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle, delimiter='\t', lineterminator='\n')
            writer.writerows(rows)
    # Strip embedded tabs/newlines to maintain two-column TSV compatibility.
    clean = lambda s: re.sub(r'[\t\r\n]+', ' ', s).strip()
    used_ids = sorted({pid for ids in candidates.values() for pid in ids})
    write('queries.tsv', ((qid, clean(query_texts[qid])) for qid in query_ids))
    write('passages.tsv', ((pid, clean(texts[pid])) for pid in used_ids))
    write('candidates.tsv', ((qid, pid) for qid in query_ids for pid in candidates[qid]))
    write('qrels.tsv', ((qid, pid, qrels[qid, pid]) for qid in query_ids for pid in candidates[qid] if (qid, pid) in qrels))
    groups = load_groups(*(output / name for name in ('queries.tsv', 'passages.tsv', 'candidates.tsv', 'qrels.tsv')))
    positives = sum(any(c.relevance is not None and c.relevance > 0 for c in g.candidates) for g in groups)
    manifest = {'dataset': 'BEIR SciFact', 'split': split, 'archive_md5': _md5(archive),
                'retriever': 'bm25-k1.2-b0.75', 'top_k': top_k,
                'query_count': len(groups), 'queries_with_retrieved_positive': positives,
                'candidate_count': sum(len(g.candidates) for g in groups),
                'max_queries': max_queries, 'positive_injection': False}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest
