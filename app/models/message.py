import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import relationship

from app.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class MessageRole:
    USER = "user"
    ASSISTANT = "assistant"


class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True, default=_uuid)
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False, index=True)

    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)

    # JSON-encoded list of citation dicts: [{id, source, page, snippet, url, chunkId, score}, ...]
    citations = Column(Text, nullable=True)
    # JSON-encoded list of document ids the *user's* question was scoped to.
    document_ids = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")
