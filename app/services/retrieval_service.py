"""Vector search over a document's (or several documents') chunks.

Two implementations, chosen by `IS_POSTGRES` (see database.py):

- Postgres: `_search_pgvector` pushes the similarity ranking down to the
  database via pgvector's `<=>` cosine-distance operator, which the HNSW
  index built in database.py's `_ensure_pgvector_index()` makes an index
  scan instead of a sequential one. This is the real path — it's what
  keeps retrieval fast as chunks grow into the hundreds of thousands or
  millions.
- SQLite (local dev / the test suite only): `_search_python_loop` is the
  original prototype approach — load every candidate chunk into Python
  and score it with cosine similarity there. Fine for a handful of
  documents; not meant to run at the scale the Postgres path targets.
"""
from __future__ import annotations

import json
import math
from typing import List, Optional, Tuple

from sqlalchemy.orm import Query, Session

from app.config import settings
from app.database import IS_POSTGRES
from app.models.chunk import Chunk
from app.models.document import Document
from app.services.embedding_service import embed_query


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def retrieve_chunks(
    db: Session,
    *,
    query: str,
    owner_id: str,
    document_ids: Optional[List[str]] = None,
    top_k: Optional[int] = None,
) -> List[Tuple[Chunk, float]]:
    """Returns [(chunk, score), ...] sorted by descending similarity, scoped
    to `document_ids` when provided (falls back to every document `owner_id`
    owns). Always constrained to `owner_id`'s own documents at the query
    level — chunks belonging to other users are never fetched as candidates,
    let alone scored or returned."""
    top_k = top_k or settings.TOP_K_RESULTS

    candidates = (
        db.query(Chunk)
        .join(Document, Chunk.document_id == Document.id)
        .filter(Document.owner_id == owner_id)
        .filter(Chunk.embedding.isnot(None))
    )
    if document_ids:
        candidates = candidates.filter(Chunk.document_id.in_(document_ids))

    query_vector = embed_query(query)

    if IS_POSTGRES:
        return _search_pgvector(candidates, query_vector, top_k)
    return _search_python_loop(candidates, query_vector, top_k)


def _search_pgvector(candidates: Query, query_vector: List[float], top_k: int) -> List[Tuple[Chunk, float]]:
    """Real vector search: orders by pgvector's `<=>` cosine-distance
    operator and lets Postgres' HNSW index (see database.py) find the
    nearest `top_k` chunks directly, instead of scanning every candidate.

    `1 - distance` converts pgvector's cosine *distance* into the cosine
    *similarity* score the rest of the app (and MIN_SIMILARITY) works in.
    That filter is applied in Python, after the index has already narrowed
    the candidates down to `top_k` — cheap at this size, and keeps the
    `ORDER BY ... LIMIT` shape that lets the HNSW index do approximate
    nearest-neighbor search efficiently (adding the threshold as a SQL
    WHERE clause on the same expression would fight the index instead).
    """
    distance = Chunk.embedding.cosine_distance(query_vector)
    rows = (
        candidates.with_entities(Chunk, distance.label("distance"))
        .order_by(distance)
        .limit(top_k)
        .all()
    )
    return [(chunk, 1 - dist) for chunk, dist in rows if (1 - dist) >= settings.MIN_SIMILARITY]


def _search_python_loop(candidates: Query, query_vector: List[float], top_k: int) -> List[Tuple[Chunk, float]]:
    """SQLite fallback (local dev / tests only) — see module docstring."""
    scored: List[Tuple[Chunk, float]] = []
    for chunk in candidates.all():
        vector = json.loads(chunk.embedding)
        score = _cosine_similarity(query_vector, vector)
        if score >= settings.MIN_SIMILARITY:
            scored.append((chunk, score))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]
