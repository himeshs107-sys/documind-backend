import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class EvaluationRun(Base):
    """A single question run through the RAG pipeline and scored, so past
    evaluation results can be listed and compared over time."""

    __tablename__ = "evaluation_runs"

    id = Column(String, primary_key=True, default=_uuid)
    owner_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)

    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)

    # JSON-encoded lists.
    citations = Column(Text, nullable=True)  # cited source names
    document_ids = Column(Text, nullable=True)
    expected_keywords = Column(Text, nullable=True)
    expected_sources = Column(Text, nullable=True)
    retrieved_chunks = Column(Text, nullable=True)  # see schemas.evaluation.RetrievedChunkOut

    keyword_coverage = Column(Float, default=1.0)
    source_coverage = Column(Float, default=1.0)
    latency_ms = Column(Integer, default=0)
    num_chunks_retrieved = Column(Integer, default=0)

    created_at = Column(DateTime, default=datetime.utcnow)

    owner = relationship("User")
