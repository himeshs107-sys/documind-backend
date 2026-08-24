"""
Turns text into vectors for semantic search.

Three providers, controlled by EMBEDDING_PROVIDER (same mock/real pattern as
the frontend's VITE_USE_MOCKS):

- "mock"  (default): a deterministic hash-based pseudo-embedding. Zero
  dependencies, same interface, good enough to exercise the whole RAG
  pipeline (upload -> chunk -> embed -> retrieve -> generate) without
  installing any ML libraries or calling out to a real model — but it has
  no actual semantic understanding (it maps shared vocabulary to nearby
  vectors, not shared meaning), so retrieval quality under this provider
  isn't representative of the real system. Don't judge retrieval/RAG
  quality against this provider; switch to "local" or "openai" first.
- "local": a real sentence-transformers model running on this machine.
  Requires `pip install -r requirements-optional.txt`. No API key or
  network access needed at query time, but you're on the hook for the
  model's memory/CPU footprint.
- "openai": OpenAI's hosted embeddings API (default model:
  text-embedding-3-small). Requires `pip install -r requirements-optional.txt`
  and OPENAI_API_KEY. The production-recommended option when you don't want
  to run embedding inference yourself — same OPENAI_API_KEY the app already
  uses for LLM_PROVIDER=openai.
"""
from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache
from typing import List

from app.config import settings

_WORD_RE = re.compile(r"[a-z0-9]+")

# text-embedding-3-small/-large support an OpenAI-specific `dimensions`
# parameter that truncates their native output to a requested size without
# a separate call — used below so a document's embedding always comes out
# at settings.EMBEDDING_DIM, matching the pgvector column width and HNSW
# index (see models/chunk.py, database.py) regardless of which provider
# produced it. Older OpenAI embedding models (e.g. text-embedding-ada-002)
# don't support this parameter — if you set EMBEDDING_MODEL_NAME to one of
# those, drop the `dimensions` kwarg below and set EMBEDDING_DIM to match
# that model's fixed output size instead.
_DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"


def _mock_embedding(text: str, dim: int) -> List[float]:
    """Deterministic pseudo-embedding: hashes each word into a signed bucket.
    Same text always maps to the same vector, and texts sharing vocabulary
    end up closer together — enough to make retrieval behave sensibly for a
    demo, without any ML dependency. Punctuation is stripped so "search?" and
    "search." both match the token "search". Not a real embedding — see the
    module docstring before treating retrieval under this provider as
    representative of the real system."""
    vector = [0.0] * dim
    words = _WORD_RE.findall(text.lower())
    if not words:
        return vector

    for word in words:
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign

    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


@lru_cache(maxsize=1)
def _get_local_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "EMBEDDING_PROVIDER=local requires sentence-transformers — "
            "pip install -r requirements-optional.txt"
        ) from exc
    return SentenceTransformer(settings.EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def _get_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "EMBEDDING_PROVIDER=openai requires the openai package — "
            "pip install -r requirements-optional.txt"
        ) from exc
    return OpenAI(api_key=settings.OPENAI_API_KEY)


def _openai_embed(texts: List[str]) -> List[List[float]]:
    client = _get_openai_client()
    # EMBEDDING_MODEL_NAME defaults to the sentence-transformers model name
    # ("all-MiniLM-L6-v2") since that's what EMBEDDING_PROVIDER=local uses
    # by default — that's not a valid OpenAI model, so fall back to a
    # sensible OpenAI default unless it's been set to an actual OpenAI
    # embedding model (e.g. "text-embedding-3-large").
    model = (
        settings.EMBEDDING_MODEL_NAME
        if settings.EMBEDDING_MODEL_NAME.startswith("text-embedding-")
        else _DEFAULT_OPENAI_EMBEDDING_MODEL
    )
    response = client.embeddings.create(model=model, input=texts, dimensions=settings.EMBEDDING_DIM)
    # The API returns embeddings in the same order as the input list, but
    # `.index` on each item guarantees it rather than assuming it.
    ordered = sorted(response.data, key=lambda item: item.index)
    return [item.embedding for item in ordered]


def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []

    if settings.EMBEDDING_PROVIDER == "local":
        model = _get_local_model()
        return [vector.tolist() for vector in model.encode(texts)]

    if settings.EMBEDDING_PROVIDER == "openai":
        return _openai_embed(texts)

    return [_mock_embedding(text, settings.EMBEDDING_DIM) for text in texts]


def embed_query(text: str) -> List[float]:
    return embed_texts([text])[0]
