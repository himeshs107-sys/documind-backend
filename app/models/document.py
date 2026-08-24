import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from app.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class DocumentStatus:
    UPLOADING = "uploading"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=_uuid)
    owner_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)

    filename = Column(String, nullable=False)  # stored filename on disk
    original_filename = Column(String, nullable=False)  # name shown to the user
    file_path = Column(String, nullable=False)
    file_size = Column(Integer, default=0)
    content_type = Column(String, default="")
    page_count = Column(Integer, default=0)

    status = Column(String, default=DocumentStatus.UPLOADING)
    error_message = Column(String, nullable=True)
    # 0-100. Driven by the background processing pipeline (extract/chunk/
    # embed/index) — see services/document_service.py:run_processing_pipeline.
    # Meaningless once status is READY (left at 100) or ERROR (left as-is).
    progress = Column(Integer, default=0)

    created_at = Column(DateTime, default=datetime.utcnow)

    owner = relationship("User", back_populates="documents")
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
