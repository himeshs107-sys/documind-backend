"""Splits extracted page text into overlapping chunks for embedding/retrieval.

Units: CHUNK_SIZE and CHUNK_OVERLAP (app/config.py) are CHARACTER counts,
not token counts. Embedding models and LLMs reason in tokens, not
characters — as a rough rule of thumb, ~4 characters ~= 1 token for English
text, so the default CHUNK_SIZE=800 is closer to ~200 tokens. Chunking is
done in characters here (instead of running a real tokenizer) to keep this
module dependency-free and fast; see config.py's CHUNK_SIZE comment if you
need exact token-budget control instead.

Structure awareness: chunk_pages() groups text into paragraph/heading/list/
code blocks (see _split_into_blocks) and packs whole blocks into each chunk
where possible, instead of slicing at a fixed character offset regardless
of what's there. A chunk boundary lands on a blank line instead of
mid-sentence, and a heading stays attached to the section that follows it
rather than ending up alone at the end of the previous chunk.

This is a heuristic over plain text, not a real document-structure parser.
pypdf and python-docx extraction (see pdf_service.py) don't preserve font
size, DOCX heading styles, or table layout, so there's no reliable signal
to build an actual Document -> Heading -> Paragraph tree from — the only
structural signals available downstream of extraction are blank lines,
list markers (-, *, 1.), indentation, and short unpunctuated lines that
read as headings. That's what the block detection below uses. Two
consequences worth knowing:

- It works best on .docx and .txt input, where extraction preserves blank
  lines between paragraphs (see pdf_service._extract_docx). pypdf's PDF
  text extraction doesn't reliably reproduce blank lines between
  paragraphs within a page, so a PDF page often still comes back as one
  long block — chunk_pages() falls back to character slicing for any
  single block bigger than CHUNK_SIZE (same as the old behavior), so
  this is never worse than before, just not always better.
- Tables aren't detected or handled specially; a table's rows are treated
  as ordinary lines and may be split across chunks like any other text.
  Recognizing tables reliably needs layout information (cell/column
  positions) that plain-text extraction has already discarded.

Getting real, layout-aware structure (true heading levels, table cells,
etc.) would mean extracting with a library that preserves formatting
(python-docx already exposes paragraph *styles*, e.g. "Heading 1" — not
used yet; pypdf would need a layout-aware alternative) and passing that
structure through instead of re-deriving it from plain text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from app.config import settings


@dataclass
class TextChunk:
    content: str
    chunk_index: int
    page_number: Optional[int]


@dataclass
class _Block:
    text: str
    kind: str  # "heading" | "list" | "code" | "paragraph"


_LIST_LINE_RE = re.compile(r"^\s{0,3}(?:[-*\u2022]|\d+[.)])\s+")
_CODE_LINE_RE = re.compile(r"^(?:\s{4,}|\t)")
_CODE_TOKENS = ("{", "}", ";", "):", "==", "->", "def ", "class ", "import ", "function ", "const ", "return ")


def _looks_like_heading(line: str) -> bool:
    """A single short line with no terminal punctuation, in a shape that
    reads as a section title: markdown-style ("# Introduction"), numbered
    ("2.1 Related Work"), ALL CAPS ("METHODOLOGY"), or short Title Case
    ("Experimental Setup"). Deliberately conservative — false negatives
    (missing a real heading) just leave a block as an ordinary paragraph,
    which is exactly the old behavior; false positives are worse, since
    they'd pull a chunk boundary somewhere it doesn't belong.
    """
    line = line.strip()
    if not line or len(line) > 90:
        return False
    if line.startswith("#"):
        return True
    if line.endswith((".", "?", "!", ",", ";", ":")):
        return False

    words = line.split()
    if not words:
        return False
    if re.match(r"^\d+(\.\d+)*[.)]?$", words[0]) and len(words) > 1:
        return True
    if line.isupper() and len(words) <= 8:
        return True
    if len(words) <= 12 and sum(1 for w in words if w[:1].isupper()) >= max(1, len(words) - 2):
        return True
    return False


def _looks_like_list_block(lines: List[str]) -> bool:
    if not lines:
        return False
    marked = sum(1 for line in lines if _LIST_LINE_RE.match(line))
    return marked >= max(1, len(lines) // 2)


def _looks_like_code_block(lines: List[str]) -> bool:
    if not lines:
        return False
    indented = sum(1 for line in lines if _CODE_LINE_RE.match(line) or not line.strip())
    punctuated = sum(1 for line in lines if any(tok in line for tok in _CODE_TOKENS))
    return indented >= max(1, len(lines) - 1) or punctuated >= max(1, len(lines) // 2)


def _split_into_blocks(text: str) -> List[_Block]:
    raw_blocks = re.split(r"\n\s*\n", text.strip())
    blocks: List[_Block] = []
    for raw in raw_blocks:
        raw = raw.strip("\n")
        if not raw.strip():
            continue
        lines = raw.split("\n")
        if len(lines) == 1 and _looks_like_heading(lines[0]):
            blocks.append(_Block(text=lines[0].strip(), kind="heading"))
        elif _looks_like_code_block(lines):
            blocks.append(_Block(text=raw.strip(), kind="code"))
        elif _looks_like_list_block(lines):
            blocks.append(_Block(text=raw.strip(), kind="list"))
        else:
            blocks.append(_Block(text=" ".join(l.strip() for l in lines if l.strip()), kind="paragraph"))
    return blocks


def _char_slice(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Fixed-size character sliding window — the original chunking
    strategy, now used only as a fallback for a single block too big to
    fit in one chunk on its own (a very long paragraph or code block), so
    nothing is ever silently dropped or produces a chunk bigger than
    chunk_size."""
    pieces: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end == len(text):
            break
        start = end - overlap if end - overlap > start else end
    return pieces


def chunk_pages(
    pages: List[Tuple[Optional[int], str]],
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> List[TextChunk]:
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP

    chunks: List[TextChunk] = []
    index = 0

    for page_number, text in pages:
        text = text.strip()
        if not text:
            continue

        blocks = _split_into_blocks(text)
        current: List[str] = []
        current_len = 0
        pending_heading: Optional[str] = None

        def flush() -> None:
            nonlocal current, current_len, index
            if not current:
                return
            piece = "\n\n".join(current).strip()
            if piece:
                chunks.append(TextChunk(content=piece, chunk_index=index, page_number=page_number))
                index += 1
            current = []
            current_len = 0

        def start_new_with_overlap(budget: int) -> None:
            """Carries the tail of the chunk just flushed into the next
            one: a trailing line boundary from the previous chunk's text,
            up to `overlap` characters, so a chunk never opens mid-word
            and retrieval near a chunk boundary still has context from
            just before it (same intent as the old implementation's
            overlap, adapted to block boundaries).

            Capped at `budget` (how much room is left for the piece that's
            about to follow) rather than always taking the full `overlap`:
            otherwise, a run of blocks each already close to chunk_size
            would carry ~overlap extra characters into every chunk after
            the first, consistently landing chunks over chunk_size instead
            of at it — e.g. an 800-char cap with 150-char overlap topping
            out at ~926 chars per chunk, not 800, for exactly that input
            shape. Capping the carried-over tail to what still fits keeps
            add_piece's chunk_size bound one it actually holds to.
            """
            nonlocal current, current_len
            if overlap <= 0 or not chunks or budget <= 0:
                return
            take = min(overlap, budget)
            tail = chunks[-1].content[-take:]
            # Trim the leading partial word (we grabbed a fixed number of
            # characters, which can start mid-word) so the carried-over
            # overlap starts cleanly instead of with a fragment like
            # "unking that...".
            if take < len(chunks[-1].content):
                parts = tail.split(None, 1)
                tail = parts[1] if len(parts) > 1 else ""
            tail = tail.strip()
            if tail:
                current.append(tail)
                current_len += len(tail)

        def add_piece(piece: str, piece_has_heading: bool = False) -> None:
            nonlocal current, current_len
            added_len = len(piece) + (2 if current else 0)  # +2 for the "\n\n" join
            if current and current_len + added_len > chunk_size:
                flush()
                # A piece that opens with a pending heading must start the
                # new chunk clean, with nothing ahead of it — that's the
                # whole point of holding the heading back in the first
                # place (see the loop below). Carrying the previous
                # chunk's overlap tail in here would put that trailing
                # fragment before the heading, leaving the heading
                # buried mid-chunk instead of leading it, which defeats
                # "a heading stays attached to the section that follows
                # it" from this module's docstring.
                if not piece_has_heading:
                    # -2 reserves room for the "\n\n" that will join the
                    # carried tail to `piece` once both are in `current`.
                    start_new_with_overlap(budget=chunk_size - len(piece) - 2)
                added_len = len(piece) + (2 if current else 0)
            current.append(piece)
            current_len += added_len

        for block in blocks:
            if block.kind == "heading":
                # Don't emit a heading as its own chunk — hold it and
                # attach it to whatever block follows, so it stays with
                # the section it introduces instead of dangling alone.
                pending_heading = block.text
                continue

            piece_has_heading = pending_heading is not None
            piece = block.text if pending_heading is None else f"{pending_heading}\n{block.text}"
            pending_heading = None

            if len(piece) > chunk_size:
                flush()
                for sliced in _char_slice(piece, chunk_size, overlap):
                    chunks.append(TextChunk(content=sliced, chunk_index=index, page_number=page_number))
                    index += 1
                continue

            add_piece(piece, piece_has_heading)

        if pending_heading is not None:
            # A heading with nothing after it on this page (e.g. it's the
            # last line) — keep it rather than silently dropping a
            # section title.
            add_piece(pending_heading, piece_has_heading=True)

        flush()

    return chunks
