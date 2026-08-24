from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class EvaluationRequest(BaseModel):
    question: str = Field(min_length=1)
    documentIds: List[str] = Field(default_factory=list)
    expectedKeywords: List[str] = Field(default_factory=list)
    expectedSources: List[str] = Field(default_factory=list)


class RetrievedChunkOut(BaseModel):
    chunkId: str
    documentName: str
    page: Optional[int] = None
    score: float
    snippet: str


class EvaluationResult(BaseModel):
    """Returned by both POST /evaluation/run (the run just performed) and
    each item in GET /evaluation/results (past runs)."""

    id: str
    question: str
    answer: str
    citations: List[str]
    retrievedChunks: List[RetrievedChunkOut]
    keywordCoverage: float  # fraction of expectedKeywords found in the answer
    sourceCoverage: float  # fraction of expectedSources present in citations
    latencyMs: int
    numChunksRetrieved: int
    createdAt: str

    model_config = ConfigDict(from_attributes=True)
