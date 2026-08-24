import uuid

from sqlalchemy import Column, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.config import settings
from app.database import IS_POSTGRES, Base

# On Postgres, Chunk.embedding is a real pgvector column so retrieval_service
# can run an indexed `<=>` similarity query (see database.py's
# _ensure_pgvector_index()) instead of pulling every chunk into Python. On
# any other dialect — SQLite, used for local dev and the test suite — pgvector
# isn't available, so we fall back to the original JSON-encoded Text column,
# and retrieval_service.py falls back to scoring it with a Python loop.
if IS_POSTGRES:
    from pgvector.sqlalchemy import Vector

    _EmbeddingType = Vector(settings.EMBEDDING_DIM)
else:
    _EmbeddingType = Text


def _uuid() -> str:
    return str(uuid.uuid4())


class Chunk(Base):
    """A slice of a document's text plus its embedding, used for retrieval."""

    __tablename__ = "chunks"

    id = Column(String, primary_key=True, default=_uuid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False, index=True)

    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=True)

    # See _EmbeddingType above: pgvector `vector(EMBEDDING_DIM)` on Postgres,
    # JSON-encoded text everywhere else. Written in document_service.py's
    # _index_document(), read in retrieval_service.py's retrieve_chunks().
    embedding = Column(_EmbeddingType, nullable=True)

    document = relationship("Document", back_populates="chunks")
