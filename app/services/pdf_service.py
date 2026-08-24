"""Extracts plain text (per page, where applicable) from uploaded documents.

PDF text extraction has two tiers:
  1. pypdf pulls the embedded text layer directly — fast, no extra deps,
     works for any PDF that has real (selectable) text.
  2. For pages where that comes back empty or too sparse to be useful
     (typically a scanned/image-only page with no text layer at all),
     and only when OCR_ENABLED=true, the page is rasterized and OCR'd
     with Tesseract as a fallback. This runs per-page rather than
     whole-document, so a mixed PDF (some real text pages, some scanned)
     only pays the OCR cost on the pages that actually need it.

OCR is opt-in (see config.py: OCR_ENABLED) because it needs system
binaries beyond what pip can install — Tesseract itself, plus Poppler for
pdf2image's PDF-to-image rendering. See requirements-optional.txt and the
README's "Enabling OCR" section for setup. With OCR_ENABLED=false (the
default), a fully scanned PDF still fails the same way it always
did — extract_pages returns pages with empty text, and
document_service._index_document turns that into a clear
"no extractable text" error instead of silently indexing zero chunks.

Tables are NOT specially handled. pypdf's extract_text() (and Tesseract's
OCR output) both flatten a table into plain text row by row, with no
signal marking where columns start or end — a table like:

    Method | Accuracy
    A      | 92%
    B      | 95%

comes out as bare lines of text with whitespace roughly where the columns
were, easy to misread as prose, or gets its column alignment destroyed
entirely if the source used space/tab-based layout rather than visible
"|" separators. Nothing downstream (chunking_service.py, embedding,
retrieval) has any cell/column structure to work with, so a chunk can
split a table mid-row, and its embedding is only as good as the flattened
text.

This is a real gap for tables in research papers, financial reports, or
any table-heavy document, and is worth fixing for a true document-
intelligence platform. It is NOT required for this project's current
(MVP) stage — flagging so it doesn't get silently forgotten, not
scheduling it. Fixing it properly means detecting table regions and
extracting them with actual layout awareness (e.g. `pdfplumber` or
`camelot` for PDFs, both of which can return a table as rows/columns
instead of flattened text) and giving chunking_service.py a distinct
"table" block kind that's chunked (or kept whole, cell structure
preserved) differently from prose — a proper fix touches this file,
chunking_service.py, and possibly the Chunk model, not just extraction.
"""
from pathlib import Path
from typing import List, Optional, Tuple

from app.config import settings


def extract_pages(file_path: str, content_type: str) -> List[Tuple[Optional[int], str]]:
    """Returns a list of (page_number, text) tuples. page_number is 1-indexed
    for formats with a real page concept (pdf); formats without one (txt,
    docx) return page_number=None for their single "page" of text, rather
    than faking a page 1 — there's no reliable page boundary to report
    (see _extract_docx below), and reporting one anyway would show a
    misleading "Page 1" on every citation from these formats, as if the
    document actually had numbered pages."""
    suffix = Path(file_path).suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf(file_path)
    if suffix == ".docx":
        return _extract_docx(file_path)
    return _extract_txt(file_path)


def _extract_pdf(file_path: str) -> List[Tuple[int, str]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pypdf is required to read PDF files — pip install pypdf") from exc

    reader = PdfReader(file_path)
    pages = []
    ocr_pages = []  # 1-indexed page numbers whose text layer was too sparse
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if len(text) < settings.OCR_MIN_CHARS_PER_PAGE:
            ocr_pages.append(i)
        if text:
            pages.append((i, text))

    if ocr_pages and settings.OCR_ENABLED:
        ocr_results = _ocr_pdf_pages(file_path, ocr_pages)
        # Replace (or add) entries for the pages OCR actually recovered
        # text from; pages where OCR also came back empty are left out,
        # same as a pypdf-empty page would be.
        by_page = {page_number: text for page_number, text in pages}
        by_page.update(ocr_results)
        pages = sorted(by_page.items())

    return pages or [(1, "")]


def _ocr_pdf_pages(file_path: str, page_numbers: List[int]) -> dict:
    """Rasterizes the given 1-indexed PDF pages and runs Tesseract OCR on
    each. Returns {page_number: text} for pages OCR found text on; pages
    OCR also couldn't read are simply absent from the result.

    Only called when OCR_ENABLED=true. Requires the `pytesseract` and
    `pdf2image` packages (requirements-optional.txt) plus the Tesseract
    and Poppler system binaries — see the backend README's "Enabling OCR"
    section for installation. A missing dependency here raises rather than
    silently skipping OCR, so misconfiguration surfaces as a clear
    document-processing error instead of documents quietly staying
    unsearchable.
    """
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError as exc:
        raise RuntimeError(
            "OCR_ENABLED=true but pytesseract/pdf2image aren't installed — "
            "pip install -r requirements-optional.txt (also requires the "
            "Tesseract and Poppler system binaries; see README.md)"
        ) from exc

    results: dict = {}
    for page_number in page_numbers:
        # first_page/last_page are 1-indexed and inclusive; rendering one
        # page at a time keeps memory bounded for large documents.
        images = convert_from_path(file_path, first_page=page_number, last_page=page_number)
        if not images:
            continue
        text = pytesseract.image_to_string(images[0], lang=settings.OCR_LANGUAGE).strip()
        if text:
            results[page_number] = text
    return results


def _extract_docx(file_path: str) -> List[Tuple[Optional[int], str]]:
    try:
        import docx  # python-docx
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("python-docx is required to read Word files — pip install python-docx") from exc

    document = docx.Document(file_path)
    # Joined with a blank line (not a single "\n") between paragraphs so
    # chunking_service.py's block splitter — which looks for blank lines as
    # its primary paragraph-boundary signal — actually gets one; see its
    # module docstring for why that signal matters more for .docx/.txt than
    # for PDFs.
    text = "\n\n".join(p.text for p in document.paragraphs if p.text.strip())
    # .docx has no reliable page boundaries without a rendering engine, so
    # its single "page" of text is reported with page_number=None rather
    # than a fake page 1 — chunking still splits it into pieces, but
    # nothing downstream should claim to know which "page" a chunk is on.
    return [(None, text)]


def _extract_txt(file_path: str) -> List[Tuple[Optional[int], str]]:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return [(None, f.read())]
