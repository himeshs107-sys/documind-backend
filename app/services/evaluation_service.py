"""
Lightweight RAG evaluation: runs a question through the real retrieval +
generation pipeline, scores it against keywords/sources you expect to see
(no labeled dataset or external eval framework required), and persists the
run so GET /evaluation/results has a history to show.
"""
from __future__ import annotations

import json
from typing import List

from sqlalchemy.orm import Session

from app.models.document import Document
from app.models.evaluation_run import EvaluationRun
from app.services.document_service import validate_owned_documents
from app.services.rag_service import generate_answer
from app.services.retrieval_service import retrieve_chunks


def _to_out_dict(run: EvaluationRun) -> dict:
    return {
        "id": run.id,
        "question": run.question,
        "answer": run.answer,
        "citations": json.loads(run.citations) if run.citations else [],
        "retrievedChunks": json.loads(run.retrieved_chunks) if run.retrieved_chunks else [],
        "keywordCoverage": run.keyword_coverage,
        "sourceCoverage": run.source_coverage,
        "latencyMs": run.latency_ms,
        "numChunksRetrieved": run.num_chunks_retrieved,
        "createdAt": run.created_at.isoformat() if run.created_at else "",
    }


def run_evaluation(
    db: Session,
    *,
    owner_id: str,
    question: str,
    document_ids: List[str],
    expected_keywords: List[str],
    expected_sources: List[str],
) -> dict:
    # Reject up front if the caller asked for a document they don't own,
    # rather than letting retrieve_chunks silently filter it out later.
    validate_owned_documents(db, owner_id=owner_id, document_ids=document_ids)

    scored_chunks = retrieve_chunks(db, query=question, owner_id=owner_id, document_ids=document_ids or None)

    # retrieve_chunks is already owner-scoped at the DB level, so this lookup
    # just hydrates the Document rows needed for citation building.
    doc_ids = {chunk.document_id for chunk, _ in scored_chunks}
    documents = db.query(Document).filter(Document.id.in_(doc_ids)).all()
    documents_by_id = {d.id: d for d in documents}

    result = generate_answer(question, scored_chunks, documents_by_id)

    answer_lower = result.text.lower()
    keyword_hits = [k for k in expected_keywords if k.lower() in answer_lower]
    keyword_coverage = (len(keyword_hits) / len(expected_keywords)) if expected_keywords else 1.0

    cited_sources = {c["source"] for c in result.citations}
    source_hits = [s for s in expected_sources if s in cited_sources]
    source_coverage = (len(source_hits) / len(expected_sources)) if expected_sources else 1.0

    retrieved_chunks_out = [
        {
            "chunkId": chunk.id,
            "documentName": documents_by_id[chunk.document_id].original_filename
            if chunk.document_id in documents_by_id
            else "unknown",
            "page": chunk.page_number,
            "score": round(score, 4),
            "snippet": chunk.content[:200],
        }
        for chunk, score in scored_chunks
    ]

    run = EvaluationRun(
        owner_id=owner_id,
        question=question,
        answer=result.text,
        citations=json.dumps([c["source"] for c in result.citations]),
        document_ids=json.dumps(document_ids) if document_ids else None,
        expected_keywords=json.dumps(expected_keywords) if expected_keywords else None,
        expected_sources=json.dumps(expected_sources) if expected_sources else None,
        retrieved_chunks=json.dumps(retrieved_chunks_out) if retrieved_chunks_out else None,
        keyword_coverage=round(keyword_coverage, 3),
        source_coverage=round(source_coverage, 3),
        latency_ms=result.latency_ms,
        num_chunks_retrieved=len(scored_chunks),
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    return _to_out_dict(run)


def list_evaluation_runs(db: Session, *, owner_id: str) -> List[dict]:
    runs = (
        db.query(EvaluationRun)
        .filter(EvaluationRun.owner_id == owner_id)
        .order_by(EvaluationRun.created_at.desc())
        .all()
    )
    return [_to_out_dict(run) for run in runs]
