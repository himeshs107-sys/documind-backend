import os
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, File, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.exceptions import NotFoundException
from app.database import get_db
from app.dependencies import get_current_user
from app.schemas.document import DocumentDeleteResponse, DocumentOut
from app.services import document_service

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentOut)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Saves the file and returns immediately with status="processing" —
    the parse/chunk/embed/index pipeline runs afterward as a background
    task, so this stays fast even for large files. Poll GET
    /documents/{id} (or GET /documents) until status is "ready" or
    "error"; `progress` (0-100) updates in the meantime.

    The upload itself is streamed straight to disk in bounded chunks (see
    document_service._stream_to_disk) with MAX_FILE_SIZE_MB enforced as
    each chunk arrives — an oversized file is rejected, and its partial
    file on disk removed, without ever buffering the whole thing into
    memory first.
    """
    document = await document_service.create_pending_document(db, owner_id=current_user.id, file=file)
    background_tasks.add_task(document_service.run_processing_pipeline, document.id)
    return DocumentOut.from_model(document)


@router.get("", response_model=List[DocumentOut])
def list_documents(db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    documents = document_service.list_documents(db, owner_id=current_user.id)
    return [DocumentOut.from_model(d) for d in documents]


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: str, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    document = document_service.get_document(db, owner_id=current_user.id, document_id=document_id)
    return DocumentOut.from_model(document)


@router.get("/{document_id}/file")
def get_document_file(document_id: str, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    """Serves the original uploaded file itself (not metadata) — this is
    what citation links (see rag_service._build_citation's "url" field) and
    the in-app PDF preview point at. GET /{document_id} above returns JSON
    describing the document; this returns the bytes.

    Owner-scoped the same way as get_document: a 404, not a 403, on a
    document that exists but belongs to someone else."""
    document = document_service.get_document(db, owner_id=current_user.id, document_id=document_id)
    if not document.file_path or not os.path.exists(document.file_path):
        raise NotFoundException("Document file not found")
    return FileResponse(
        document.file_path,
        media_type=document.content_type or "application/octet-stream",
        filename=document.original_filename,
    )


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
def delete_document(document_id: str, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    document_service.delete_document(db, owner_id=current_user.id, document_id=document_id)
    return DocumentDeleteResponse(id=document_id, deleted=True)
