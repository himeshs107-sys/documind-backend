"""Document upload, listing, retrieval, and deletion — including the
parse -> chunk -> embed -> index pipeline that makes a document searchable.

The pipeline runs out-of-request as a FastAPI BackgroundTask (see
app/api/documents.py): the upload endpoint saves the file, inserts a
Document row with status="processing", and returns immediately. The client
polls GET /documents/{id} (or GET /documents) until status flips to "ready"
or "error", using `progress` (0-100) to render a progress bar in the
meantime. This keeps big files (100MB PDFs, 500-page textbooks) from tying
up the upload request long enough to hit a client/proxy timeout.

BackgroundTasks run in this same process after the response is sent, so
they're a good fit at this app's scale without adding infra (Celery+Redis,
etc.) — see database.py's docstring for the same philosophy applied to
SQLite vs. Postgres. The tradeoff: a BackgroundTask is pure in-process
state, not a durable job — if the server crashes or restarts mid-pipeline,
that job is simply gone, with nothing left to resume it. There's no queue
to replay from. recover_interrupted_documents() (below) doesn't change
that; it only makes the failure visible (flags the stuck document as
"error" at the next startup) instead of leaving it silently stuck at
"processing" forever. If processing needs to actually survive a crash, or
run on a separate worker fleet, swap run_processing_pipeline's call site
for a real task queue (Celery+Redis or similar) — nothing else here would
need to change. Deliberately not doing that yet at this project's current
stage; see README.md "Background processing & durability".
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import List

from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.config import settings
from app.core.exceptions import FileTooLargeException, NotFoundException, UnsupportedFileTypeException
from app.database import IS_POSTGRES, SessionLocal
from app.models.chunk import Chunk
from app.models.document import Document, DocumentStatus
from app.services import chunking_service, pdf_service
from app.services.embedding_service import embed_texts
from app.utils.file_utils import ensure_upload_dir, safe_filename

# Coarse progress checkpoints for the four pipeline stages. Not meant to be
# precise — just enough for the UI to show visible movement instead of a
# bar that sits at 0% for a long extract/embed step then jumps to 100%.
_PROGRESS_SAVED = 5
_PROGRESS_EXTRACTED = 35
_PROGRESS_CHUNKED = 50
_PROGRESS_EMBEDDED = 90
_PROGRESS_DONE = 100


_UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1MiB per read/write — caps how much of an
# in-flight upload sits in memory at once, regardless of the file's total
# size, while still being large enough that per-chunk overhead is negligible.


def _validate_extension(file: UploadFile) -> None:
    """Checked before a single byte of the upload is read, so an
    unsupported extension is rejected without touching the network stream
    or the disk at all."""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in settings.allowed_extensions_list:
        raise UnsupportedFileTypeException(
            f"'{ext}' isn't supported. Allowed types: {', '.join(settings.allowed_extensions_list)}"
        )


async def _stream_to_disk(file: UploadFile, dest_path: str) -> int:
    """Copies `file` to `dest_path` in fixed-size chunks, enforcing
    MAX_FILE_SIZE_MB as each chunk arrives instead of buffering the whole
    upload into memory first with `await file.read()`. A file over the
    limit is caught as soon as the running total crosses it — the rest of
    the upload is never read, and the partial file already written to disk
    is deleted — rather than after the entire (potentially huge) upload has
    already landed in memory. Returns the final size in bytes.
    """
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    total = 0
    try:
        with open(dest_path, "wb") as out:
            while True:
                chunk = await file.read(_UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise FileTooLargeException(f"File exceeds the {settings.MAX_FILE_SIZE_MB}MB limit")
                out.write(chunk)
    except Exception:
        # Don't leave a truncated, half-written file behind — whether the
        # failure was the size check above or a disk error mid-write.
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise
    return total


async def create_pending_document(db: Session, *, owner_id: str, file: UploadFile) -> Document:
    """Validates and saves the upload, then inserts the Document row with
    status="processing" and returns immediately. The slow parse/chunk/embed
    work happens afterward in run_processing_pipeline — call that as a
    BackgroundTask right after this in the same request (see
    api/documents.py) so it starts as soon as the upload is on disk.

    The upload itself is streamed straight to its final destination on disk
    in bounded chunks (see _stream_to_disk) rather than read fully into
    memory first — an oversized upload is caught and its partial file
    removed mid-stream, before the whole thing has to sit in RAM at once.
    """
    _validate_extension(file)

    upload_dir = ensure_upload_dir(settings.UPLOAD_DIR)
    stored_name = f"{uuid.uuid4()}_{safe_filename(file.filename or 'document')}"
    file_path = os.path.join(upload_dir, stored_name)

    size = await _stream_to_disk(file, file_path)

    document = Document(
        owner_id=owner_id,
        filename=stored_name,
        original_filename=file.filename or stored_name,
        file_path=file_path,
        file_size=size,
        content_type=file.content_type or "",
        status=DocumentStatus.PROCESSING,
        progress=_PROGRESS_SAVED,
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    return document


def run_processing_pipeline(document_id: str) -> None:
    """The background task. Runs after the upload response has already been
    sent, so it opens its own DB session rather than reusing the
    request-scoped one (which is closed by the time this runs).

    Two different moments a concurrent delete_document() call can land,
    both handled, neither needing this function to re-check anything
    mid-flight itself:
    - Before the fetch below: `document` comes back None and this returns
      immediately — nothing was ever loaded, nothing to clean up.
    - After the fetch below, anywhere before the final commit: this
      function has no idea the row is gone and keeps going regardless —
      the Chunk rows it eventually tries to persist reference a document_id
      that no longer exists. Postgres' foreign key constraint rejects that
      insert; database.py's `_enable_sqlite_foreign_keys` pragma makes
      SQLite do the same (it doesn't enforce FKs by default). Either way
      the resulting IntegrityError lands in the except block below, which
      rolls back, re-fetches (finds nothing), and exits quietly.
    """
    db = SessionLocal()
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        if document is None:
            return  # deleted before processing started

        try:
            _index_document(db, document)
            document.status = DocumentStatus.READY
            document.progress = _PROGRESS_DONE
            db.commit()
        except Exception as exc:  # noqa: BLE001 — surface any parsing/embedding failure on the record
            # Discard any partially-added Chunk rows from the failed attempt
            # (e.g. embedding succeeded but a later chunk write raised) so we
            # don't persist an inconsistent half-indexed document. Rollback
            # expires `document`, so re-fetch it before setting the error.
            db.rollback()
            document = db.query(Document).filter(Document.id == document_id).first()
            if document is not None:
                document.status = DocumentStatus.ERROR
                document.error_message = str(exc)[:500]
                db.commit()
    finally:
        db.close()


def _index_document(db: Session, document: Document) -> None:
    pages = pdf_service.extract_pages(document.file_path, document.content_type)
    document.page_count = len(pages)
    document.progress = _PROGRESS_EXTRACTED
    db.commit()

    text_chunks = chunking_service.chunk_pages(pages)
    document.progress = _PROGRESS_CHUNKED
    db.commit()

    if not text_chunks:
        # Previously this just returned, silently leaving the document
        # "ready" with zero chunks — indistinguishable in the UI from an
        # empty-but-successfully-parsed file. Raising here means it goes
        # through the normal ERROR path in run_processing_pipeline, so the
        # user actually sees why (see DocumentOut.errorMessage).
        if Path(document.file_path).suffix.lower() == ".pdf" and not settings.OCR_ENABLED:
            raise ValueError(
                "No extractable text found — this PDF may be scanned or image-based. "
                "Enable OCR_ENABLED to process scanned PDFs."
            )
        raise ValueError("No extractable text found in this document.")

    vectors = embed_texts([c.content for c in text_chunks])
    document.progress = _PROGRESS_EMBEDDED
    db.commit()

    for text_chunk, vector in zip(text_chunks, vectors):
        db.add(
            Chunk(
                document_id=document.id,
                content=text_chunk.content,
                chunk_index=text_chunk.chunk_index,
                page_number=text_chunk.page_number,
                # pgvector accepts a plain list directly on Postgres; the
                # SQLite fallback column is JSON-encoded text (see
                # models/chunk.py and retrieval_service.py).
                embedding=vector if IS_POSTGRES else json.dumps(vector),
            )
        )


def recover_interrupted_documents(db: Session) -> int:
    """Best-effort mitigation for the sharpest edge of using BackgroundTasks
    as a job runner: nothing about it survives the process dying. If the
    server crashes or restarts while run_processing_pipeline() is mid-flight
    for some document, that document's row is left sitting at
    status="processing" forever — no error, no retry, just a progress bar
    that never moves again and no indication to the user why.

    This can't recover the lost work — BackgroundTasks has no persistence
    to recover it *from*, which is exactly why it isn't a real job queue
    (see this module's docstring and README.md's "Background processing &
    durability"). What it does is make the failure visible instead of
    silent: called once at startup (see app/main.py), it flags any document
    still marked "processing" as an error the user can act on (re-upload)
    instead of a stuck spinner with no explanation.

    Only correct for a single-process deployment — which BackgroundTasks
    already assumes, since a task only ever runs in the process that
    scheduled it. Nothing can legitimately still be "processing" from a
    previous run when this process is only just starting up. Running
    multiple app instances behind a load balancer breaks that assumption
    (this could flag a document a *different*, still-running instance is
    actively processing) — one more reason this approach doesn't scale past
    a single instance. Move to Celery + Redis (or another durable queue)
    once that's a problem, and delete this function when you do: a durable
    queue's jobs survive a crash and resume on their own, so there's
    nothing left here to reconcile at startup.
    """
    interrupted = db.query(Document).filter(Document.status == DocumentStatus.PROCESSING).all()
    for document in interrupted:
        document.status = DocumentStatus.ERROR
        document.error_message = (
            "Processing was interrupted, likely by a server restart, before this document "
            "finished indexing. Please re-upload it."
        )
    if interrupted:
        db.commit()
    return len(interrupted)


def list_documents(db: Session, *, owner_id: str) -> List[Document]:
    return (
        db.query(Document)
        .filter(Document.owner_id == owner_id)
        .order_by(Document.created_at.desc())
        .all()
    )


def get_document(db: Session, *, owner_id: str, document_id: str) -> Document:
    doc = db.query(Document).filter(Document.id == document_id, Document.owner_id == owner_id).first()
    if not doc:
        raise NotFoundException("Document not found")
    return doc


def validate_owned_documents(db: Session, *, owner_id: str, document_ids: List[str]) -> None:
    """Rejects up front if any of `document_ids` doesn't exist or isn't owned
    by `owner_id`, instead of letting retrieval silently drop it later. Used
    to check a request's requested document scope (e.g. chat's documentIds,
    evaluation's documentIds) before any RAG work runs, not just before the
    final citation/document hydration lookup.

    Raises NotFoundException (404, not 403) on the first mismatch — same as
    get_document() above — so a non-owner learns nothing about whether the
    ID exists at all vs. exists-but-belongs-to-someone-else.
    """
    if not document_ids:
        return
    owned_count = (
        db.query(Document)
        .filter(Document.id.in_(document_ids), Document.owner_id == owner_id)
        .count()
    )
    if owned_count != len(set(document_ids)):
        raise NotFoundException("One or more documents not found")


def delete_document(db: Session, *, owner_id: str, document_id: str) -> None:
    doc = get_document(db, owner_id=owner_id, document_id=document_id)
    if doc.file_path and os.path.exists(doc.file_path):
        try:
            os.remove(doc.file_path)
        except OSError:
            pass
    db.delete(doc)
    db.commit()
