import io

import pytest


def _upload_txt(
    client,
    headers,
    content=b"A* is an informed search algorithm that uses a heuristic function.",
    filename="notes.txt",
):
    return client.post(
        "/api/documents/upload",
        headers=headers,
        files={"file": (filename, io.BytesIO(content), "text/plain")},
    )


def test_upload_requires_authentication(client):
    response = client.post(
        "/api/documents/upload", files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")}
    )
    assert response.status_code == 401


def test_upload_txt_document_returns_processing_immediately(client, auth_headers):
    # The upload response comes back before the parse/chunk/embed pipeline
    # runs — status starts "processing", not "ready".
    response = _upload_txt(client, auth_headers)
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "notes.txt"
    assert body["status"] == "processing"
    assert body["size"] > 0
    assert body["uploadedAt"]


def test_document_becomes_ready_after_background_processing(client, auth_headers):
    # TestClient runs BackgroundTasks synchronously before the upload call
    # returns, so by the time we poll, the pipeline has already finished.
    upload = _upload_txt(client, auth_headers, filename="processed.txt")
    doc_id = upload.json()["id"]

    response = client.get(f"/api/documents/{doc_id}", headers=auth_headers)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["progress"] == 100


def test_upload_rejects_unsupported_extension(client, auth_headers):
    response = client.post(
        "/api/documents/upload",
        headers=auth_headers,
        files={"file": ("notes.xyz", io.BytesIO(b"hello"), "application/octet-stream")},
    )
    assert response.status_code == 400


def test_list_documents_returns_uploaded_files(client, auth_headers):
    _upload_txt(client, auth_headers, filename="a.txt")
    _upload_txt(client, auth_headers, filename="b.txt")

    response = client.get("/api/documents", headers=auth_headers)
    assert response.status_code == 200
    names = [d["name"] for d in response.json()]
    assert "a.txt" in names
    assert "b.txt" in names


def test_get_single_document(client, auth_headers):
    upload = _upload_txt(client, auth_headers, filename="single.txt")
    doc_id = upload.json()["id"]

    response = client.get(f"/api/documents/{doc_id}", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["name"] == "single.txt"


def test_get_document_not_found(client, auth_headers):
    response = client.get("/api/documents/does-not-exist", headers=auth_headers)
    assert response.status_code == 404


def test_delete_document_removes_it(client, auth_headers):
    upload = _upload_txt(client, auth_headers, filename="to_delete.txt")
    doc_id = upload.json()["id"]

    delete_response = client.delete(f"/api/documents/{doc_id}", headers=auth_headers)
    assert delete_response.status_code == 200
    assert delete_response.json()["deleted"] is True

    get_response = client.get(f"/api/documents/{doc_id}", headers=auth_headers)
    assert get_response.status_code == 404


def test_deleted_document_rejects_orphaned_chunk_inserts(client, auth_headers):
    """Regression test for the race run_processing_pipeline has to survive:
    a user can delete a document *after* the background pipeline has
    already fetched it and started working (run_processing_pipeline's own
    "document is None" check only catches a delete landing *before* that
    initial fetch — see its docstring), so the pipeline can still reach
    the point of trying to persist Chunk rows referencing a document_id
    that no longer exists.

    Postgres already rejects that insert via its always-on foreign key
    constraint — an IntegrityError that run_processing_pipeline's existing
    broad except block already handles correctly (rolls back, re-fetches
    the document, finds nothing, and exits quietly with no error written).
    SQLite doesn't enforce foreign keys by default, so without
    database.py's `_enable_sqlite_foreign_keys` pragma, this same insert
    used to succeed anyway — leaving orphaned Chunk rows behind forever,
    silently, with no error and no visible symptom.
    """
    from sqlalchemy.exc import IntegrityError

    from app.database import SessionLocal
    from app.models.chunk import Chunk
    from app.models.document import Document

    upload = _upload_txt(client, auth_headers, filename="raced.txt")
    doc_id = upload.json()["id"]

    db = SessionLocal()
    try:
        # TestClient runs BackgroundTasks synchronously, so by this point
        # run_processing_pipeline has already finished for real and this
        # document already has its own genuine Chunk rows. The race this
        # test simulates is supposed to land *before* those inserts land
        # (see docstring), so clear them out first -- otherwise the raw
        # delete below hits the FK constraint on those pre-existing rows
        # immediately, instead of on the orphaned insert we're trying to
        # exercise.
        db.query(Chunk).filter(Chunk.document_id == doc_id).delete()
        db.commit()

        # A raw delete of just the row — standing in for the moment
        # delete_document() commits, on its own separate session/request,
        # while the background pipeline (elsewhere, mid-flight) still
        # holds this document_id and is about to insert Chunk rows for it.
        db.query(Document).filter(Document.id == doc_id).delete()
        db.commit()

        db.add(Chunk(document_id=doc_id, content="orphaned", chunk_index=0, page_number=1))
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()

        orphaned = db.query(Chunk).filter(Chunk.document_id == doc_id).count()
        assert orphaned == 0
    finally:
        db.close()


def _blank_pdf_bytes() -> bytes:
    """A syntactically valid single-page PDF with no text layer at all —
    stands in for a scanned/image-only page without needing a binary test
    fixture or real OCR: pypdf extracts zero characters from it either way."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_scanned_pdf_without_ocr_fails_with_clear_error(client, auth_headers):
    # OCR_ENABLED=false (the test/dev default) — a PDF with no extractable
    # text should fail loudly instead of silently indexing as an empty,
    # unsearchable document (see document_service._index_document).
    response = client.post(
        "/api/documents/upload",
        headers=auth_headers,
        files={"file": ("scanned.pdf", io.BytesIO(_blank_pdf_bytes()), "application/pdf")},
    )
    doc_id = response.json()["id"]

    result = client.get(f"/api/documents/{doc_id}", headers=auth_headers).json()
    assert result["status"] == "error"
    assert "extractable text" in result["errorMessage"]
    assert "OCR" in result["errorMessage"]


def test_scanned_pdf_with_ocr_enabled_uses_ocr_fallback(client, auth_headers, monkeypatch):
    # Exercise the OCR code path without depending on the real Tesseract/
    # Poppler binaries being installed in the test environment — mock the
    # two OCR calls pdf_service._ocr_pdf_pages makes (imported inline,
    # inside the function, specifically so tests can patch sys.modules
    # like this) and assert the result flows through to a real chunk/
    # embedding, ending with the document "ready".
    import sys

    from app.config import settings

    monkeypatch.setattr(settings, "OCR_ENABLED", True)

    class _FakeImage:
        pass

    def fake_convert_from_path(file_path, first_page, last_page):
        assert first_page == last_page == 1
        return [_FakeImage()]

    def fake_image_to_string(image, lang):
        assert isinstance(image, _FakeImage)
        assert lang == settings.OCR_LANGUAGE
        return "Text recovered via OCR from a scanned page."

    fake_pdf2image = type("FakePdf2Image", (), {"convert_from_path": staticmethod(fake_convert_from_path)})
    fake_pytesseract = type("FakePytesseract", (), {"image_to_string": staticmethod(fake_image_to_string)})
    monkeypatch.setitem(sys.modules, "pdf2image", fake_pdf2image)
    monkeypatch.setitem(sys.modules, "pytesseract", fake_pytesseract)

    response = client.post(
        "/api/documents/upload",
        headers=auth_headers,
        files={"file": ("scanned.pdf", io.BytesIO(_blank_pdf_bytes()), "application/pdf")},
    )
    doc_id = response.json()["id"]

    result = client.get(f"/api/documents/{doc_id}", headers=auth_headers).json()
    assert result["status"] == "ready"
    assert result["progress"] == 100
