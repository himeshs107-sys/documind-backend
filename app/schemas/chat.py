"""
Matches the shapes frontend/src/services/chatApi.js sends and expects.
In particular, CitationOut's fields (id, source, page, snippet, url) match
exactly what the frontend's citations/ components (Citation, SourceCard,
SourcePreview) read, and MessageOut.text is expected to contain inline
{{n}} markers that frontend/src/utils/citations.js parses.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class CitationOut(BaseModel):
    id: str
    source: str
    page: Optional[int] = None
    snippet: str
    url: str = "#"


class ConversationOut(BaseModel):
    id: str
    title: str
    createdAt: str

    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def from_model(cls, conv) -> "ConversationOut":
        return cls(
            id=conv.id,
            title=conv.title,
            createdAt=conv.created_at.isoformat() if conv.created_at else "",
        )


class MessageOut(BaseModel):
    id: str
    role: str
    text: str
    citations: List[CitationOut] = Field(default_factory=list)
    createdAt: str

    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def from_model(cls, msg, citations: Optional[List[CitationOut]] = None) -> "MessageOut":
        return cls(
            id=msg.id,
            role=msg.role,
            text=msg.content,
            citations=citations or [],
            createdAt=msg.created_at.isoformat() if msg.created_at else "",
        )


class SendMessageRequest(BaseModel):
    text: str = Field(min_length=1)
    documentIds: List[str] = Field(default_factory=list)


class RegenerateRequest(BaseModel):
    userText: str = Field(min_length=1)
    # Deprecated / ignored server-side: regenerate_message(_stream) in
    # app/api/chat.py now derives the document scope from the original
    # question's stored Message.document_ids instead of trusting whatever
    # the client sends here, so an old answer can't get silently re-scoped
    # to whatever documents happen to be checked in the sidebar at
    # regenerate time. Kept only so older frontend builds that still send
    # it don't fail request validation.
    documentIds: List[str] = Field(default_factory=list)
