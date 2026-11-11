"""TSV query/passage groups with explicit relevance-judgment status.

Expected UTF-8, tab-separated files without headers:
  queries:    query_id TAB query_text
  passages:   passage_id TAB passage_text
  candidates: query_id TAB passage_id
  qrels:      query_id TAB passage_id TAB integer_relevance (>= 0)

A candidate missing from qrels is unjudged, NOT a negative example.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path


@dataclass(frozen=True)
class Candidate:
    passage_id: str
    passage: str
    relevance: int | None  # None means unjudged; 0 means judged nonrelevant


@dataclass(frozen=True)
class QueryGroup:
    query_id: str
    query: str
    candidates: tuple[Candidate, ...]


def _read_tsv(path: str | Path, columns: int):
    with Path(path).open('r', encoding='utf-8', newline='') as handle:
        for line_no, raw in enumerate(handle, 1):
            line = raw.rstrip('\r\n')
            if not line.strip():
                continue
            fields = line.split('\t')
            if len(fields) != columns or any(not f.strip() for f in fields):
                raise ValueError(f'{path}:{line_no}: expected {columns} nonempty TSV columns')
            yield fields


def _unique_text(path: str | Path) -> dict[str, str]:
    result = {}
    for key, value in _read_tsv(path, 2):
        if key in result:
            raise ValueError(f'{path}: duplicate identifier {key!r}')
        result[key] = value
    return result


def load_groups(queries: str | Path, passages: str | Path,
                candidates: str | Path, qrels: str | Path) -> tuple[QueryGroup, ...]:
    """Load groups, retaining judged zeros and unjudged candidates separately.

    Qrels may include judged query/passage pairs not in the candidate pool;
    these are retained only as judgments, not added to the candidate list.
    """
    query_text = _unique_text(queries)
    passage_text = _unique_text(passages)
    if not query_text or not passage_text:
        raise ValueError('queries and passages must be nonempty')

    judgments = {}
    for qid, pid, raw in _read_tsv(qrels, 3):
        if qid not in query_text or pid not in passage_text:
            raise ValueError('qrels reference unknown query or passage')
        try:
            relevance = int(raw)
        except ValueError as exc:
            raise ValueError('qrels relevance must be a nonnegative integer') from exc
        if relevance < 0 or (qid, pid) in judgments:
            raise ValueError('negative or duplicate qrels entry')
        judgments[(qid, pid)] = relevance

    grouped = {qid: [] for qid in query_text}
    seen = set()
    for qid, pid in _read_tsv(candidates, 2):
        if qid not in query_text or pid not in passage_text:
            raise ValueError('candidates reference unknown query or passage')
        if (qid, pid) in seen:
            raise ValueError('duplicate query/passage candidate')
        seen.add((qid, pid))
        grouped[qid].append(Candidate(pid, passage_text[pid], judgments.get((qid, pid))))

    if not seen:
        raise ValueError('candidate pool must be nonempty')
    return tuple(QueryGroup(qid, text, tuple(grouped[qid]))
                 for qid, text in query_text.items() if grouped[qid])


def split_groups(groups: tuple[QueryGroup, ...], *, test_fraction: float = 0.2,
                 seed: int = 0) -> tuple[tuple[QueryGroup, ...], tuple[QueryGroup, ...]]:
    """Deterministic query-disjoint split, independent of input row order.

    A stable hash ranks query IDs, then assigns exactly round(N * fraction)
    groups to test (clamped to ensure both splits are nonempty).
    """
    if not 0 < test_fraction < 1 or not isinstance(seed, int) or seed < 0:
        raise ValueError('invalid test fraction or seed')
    if len(groups) < 2:
        raise ValueError('at least two query groups are needed for a split')
    ids = [g.query_id for g in groups]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate query IDs')
    count = min(len(groups) - 1, max(1, round(len(groups) * test_fraction)))
    ranked = sorted(ids, key=lambda qid: (hashlib.sha256(
        f'{seed}:{qid}'.encode('utf-8')).digest(), qid))
    test_ids = set(ranked[:count])
    train = tuple(g for g in groups if g.query_id not in test_ids)
    test = tuple(g for g in groups if g.query_id in test_ids)
    return train, test


def judgment_coverage(groups: tuple[QueryGroup, ...]) -> dict[str, int]:
    """Count candidate judgments; no assumptions about missing qrels."""
    all_candidates = [c for group in groups for c in group.candidates]
    return {
        'candidates': len(all_candidates),
        'judged_relevant': sum(c.relevance is not None and c.relevance > 0 for c in all_candidates),
        'judged_nonrelevant': sum(c.relevance == 0 for c in all_candidates),
        'unjudged': sum(c.relevance is None for c in all_candidates),
    }
