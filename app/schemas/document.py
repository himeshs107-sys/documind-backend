"""Matches the shape frontend/src/services/documentApi.js expects back."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class DocumentOut(BaseModel):
    id: str
    name: str
    size: int
    status: str
    progress: int = 0
    uploadedAt: str
    errorMessage: str | None = None

    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def from_model(cls, doc) -> "DocumentOut":
        return cls(
            id=doc.id,
            name=doc.original_filename,
            size=doc.file_size or 0,
            status=doc.status,
            progress=doc.progress or 0,
            uploadedAt=doc.created_at.isoformat() if doc.created_at else "",
            errorMessage=doc.error_message,
        )


class DocumentDeleteResponse(BaseModel):
    id: str
    deleted: bool = True
