"""
Generates an answer from retrieved chunks, in both mock and real-LLM modes.

Answers are formatted with inline {{n}} citation markers — this matches the
convention the DocuMind frontend already parses (see the frontend's
src/utils/citations.js parseAnswerSegments()), so the two are wire-compatible
with no translation layer needed on either side.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Tuple

from app.config import settings
from app.models.chunk import Chunk
from app.models.document import Document
from app.models.message import Message

ScoredChunk = Tuple[Chunk, float]
TokenStream = Generator[str, None, None]
HISTORY_TURNS_FOR_REWRITE = 6  # last N messages (user+assistant) fed to the rewriter


@dataclass
class GeneratedAnswer:
    text: str
    citations: List[dict] = field(default_factory=list)
    latency_ms: int = 0


def _format_history(history: List[Message]) -> str:
    return "\n".join(f"{m.role}: {m.content}" for m in history)


def _mock_rewrite(question: str, history: List[Message]) -> str:
    """Heuristic fallback used in mock mode / when no LLM provider is configured:
    if the question looks like a short follow-up (pronouns, no nouns of its own),
    fold in the previous user question so retrieval has something concrete to embed."""
    if not history:
        return question

    pronoun_markers = ("it", "that", "this", "them", "those", "these")
    lowered = question.lower().strip()
    looks_like_followup = len(question.split()) <= 8 and any(
        f" {p} " in f" {lowered} " or lowered.startswith(p + " ") for p in pronoun_markers
    )
    if not looks_like_followup:
        return question

    last_user = next((m.content for m in reversed(history) if m.role == "user"), None)
    if not last_user:
        return question
    return f"{last_user} — {question}"


def _llm_rewrite(question: str, history: List[Message]) -> str:
    """Uses the configured LLM to turn a context-dependent follow-up into a
    standalone query, e.g. 'explain it with an example' + history about A*
    becomes 'explain the A* algorithm with an example'."""
    system = (
        "Rewrite the user's latest message into a standalone search query that "
        "captures its full meaning without needing the conversation for context. "
        "Resolve pronouns and implicit references using the conversation history. "
        "If the message is already standalone, return it unchanged. "
        "Respond with ONLY the rewritten query, no preamble or explanation."
    )
    prompt = f"Conversation history:\n{_format_history(history)}\n\nLatest message: {question}"

    try:
        if settings.LLM_PROVIDER == "openai":
            from openai import OpenAI

            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            )
            rewritten = (response.choices[0].message.content or "").strip()
        elif settings.LLM_PROVIDER == "anthropic":
            import anthropic

            client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
            response = client.messages.create(
                model=settings.LLM_MODEL,
                max_tokens=200,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            rewritten = "".join(block.text for block in response.content if hasattr(block, "text")).strip()
        else:
            return _mock_rewrite(question, history)
    except Exception:
        # Rewriting is a best-effort enhancement — never let it break retrieval.
        return _mock_rewrite(question, history)

    return rewritten or question


def rewrite_query(question: str, history: List[Message]) -> str:
    """Folds recent conversation context into `question` so retrieval embeds a
    self-contained query instead of a bare pronoun-laden follow-up. `history`
    should be the most recent turns, oldest first, NOT including `question` itself."""
    if not history:
        return question
    recent = history[-HISTORY_TURNS_FOR_REWRITE:]
    if settings.LLM_PROVIDER in ("openai", "anthropic"):
        return _llm_rewrite(question, recent)
    return _mock_rewrite(question, recent)


def _build_citation(chunk: Chunk, document: Document, score: float) -> dict:
    snippet = chunk.content.strip()
    if len(snippet) > 280:
        snippet = snippet[:280].rsplit(" ", 1)[0] + "…"

    url = f"/api/documents/{document.id}/file"
    # #page=N is a PDF.js/native-viewer convention (only meaningful for a
    # PDF); DOCX/TXT have no equivalent in-file page anchor, and
    # chunk.page_number is None for those formats (see pdf_service.py's
    # extract_pages docstring), so there's nothing to attach here anyway —
    # the `and chunk.page_number` guard below is what actually skips it.
    # Checked by extension rather than the client-supplied content_type,
    # since that MIME type isn't guaranteed accurate — same reasoning as
    # pdf_service.extract_pages's own suffix-based dispatch.
    if document.original_filename.lower().endswith(".pdf") and chunk.page_number:
        url += f"#page={chunk.page_number}"

    return {
        "id": f"cit-{chunk.id}",
        "source": document.original_filename,
        "page": chunk.page_number,
        "snippet": snippet,
        "url": url,
        "chunkId": chunk.id,
        "score": round(score, 4),
    }


def _relevant_and_citations(
    scored_chunks: List[ScoredChunk], documents_by_id: Dict[str, Document]
) -> Tuple[List[ScoredChunk], List[dict]]:
    """Filters retrieved chunks down to ones whose document is in scope, and
    builds their citation objects. Citations only depend on retrieval, not
    generation, so both the non-streaming and streaming paths compute them
    up front — the streaming path can hand them to the client immediately,
    before a single token of the answer itself exists."""
    relevant = [(c, s) for c, s in scored_chunks if c.document_id in documents_by_id]
    citations = [_build_citation(chunk, documents_by_id[chunk.document_id], score) for chunk, score in relevant]
    return relevant, citations


def _build_context(relevant: List[ScoredChunk], documents_by_id: Dict[str, Document]) -> str:
    def _label(chunk: Chunk, document: Document) -> str:
        # page_number is None for DOCX/TXT (no reliable page concept — see
        # pdf_service.extract_pages) — omit the page clause entirely rather
        # than feeding the model a literal "page None", which reads like a
        # real citation detail and invites it to repeat that nonsense back
        # in the answer.
        if chunk.page_number is None:
            return f"({document.original_filename})"
        return f"({document.original_filename}, page {chunk.page_number})"

    return "\n\n".join(
        f"[{i + 1}] {_label(chunk, documents_by_id[chunk.document_id])}: {chunk.content}"
        for i, (chunk, _score) in enumerate(relevant)
    )


def _mock_answer(question: str, scored_chunks: List[ScoredChunk]) -> str:
    if not scored_chunks:
        return (
            f'I couldn\'t find anything in your selected documents about "{question}". '
            "Try selecting different documents, or upload one that covers this topic."
        )

    lead = scored_chunks[0][0].content.strip().split(". ")[0].strip()
    if lead and not lead.endswith((".", "!", "?")):
        lead += "."
    lead_sentence = (lead[0].lower() + lead[1:]) if lead else lead

    parts = [f"Based on your documents, {lead_sentence} {{{{1}}}}"]
    if len(scored_chunks) > 1:
        parts.append("This is supported by additional context from your other selected documents. {{2}}")
    parts.append(
        'Set LLM_PROVIDER to "openai" or "anthropic" in your .env for real generated answers '
        "instead of this mock response."
    )
    return " ".join(parts)


def _openai_answer(question: str, context: str, history: List[Message]) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("LLM_PROVIDER=openai requires the openai package — pip install openai") from exc

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    system = (
        "You are DocuMind, a RAG assistant. Answer using ONLY the numbered context passages "
        "provided. Cite claims inline with {{n}} matching the passage number, e.g. 'X causes Y {{1}}'. "
        "If the context doesn't answer the question, say so. Use the prior conversation only to "
        "understand what the user means, not as a source of facts."
    )
    messages = [{"role": "system", "content": system}]
    for m in history[-HISTORY_TURNS_FOR_REWRITE:]:
        messages.append({"role": "user" if m.role == "user" else "assistant", "content": m.content})
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})

    response = client.chat.completions.create(model=settings.LLM_MODEL, messages=messages)
    return response.choices[0].message.content or ""


def _anthropic_answer(question: str, context: str, history: List[Message]) -> str:
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "LLM_PROVIDER=anthropic requires the anthropic package — pip install anthropic"
        ) from exc

    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    system = (
        "You are DocuMind, a RAG assistant. Answer using ONLY the numbered context passages "
        "provided. Cite claims inline with {{n}} matching the passage number, e.g. 'X causes Y {{1}}'. "
        "If the context doesn't answer the question, say so. Use the prior conversation only to "
        "understand what the user means, not as a source of facts."
    )
    messages = [{"role": "user" if m.role == "user" else "assistant", "content": m.content} for m in history[-HISTORY_TURNS_FOR_REWRITE:]]
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})

    response = client.messages.create(model=settings.LLM_MODEL, max_tokens=1024, system=system, messages=messages)
    return "".join(block.text for block in response.content if hasattr(block, "text"))


def generate_answer(
    question: str,
    scored_chunks: List[ScoredChunk],
    documents_by_id: Dict[str, Document],
    history: List[Message] | None = None,
) -> GeneratedAnswer:
    history = history or []
    start = time.perf_counter()

    relevant, citations = _relevant_and_citations(scored_chunks, documents_by_id)

    if settings.LLM_PROVIDER == "mock" or not relevant:
        text = _mock_answer(question, relevant)
    else:
        context = _build_context(relevant, documents_by_id)
        if settings.LLM_PROVIDER == "openai":
            text = _openai_answer(question, context, history)
        elif settings.LLM_PROVIDER == "anthropic":
            text = _anthropic_answer(question, context, history)
        else:
            text = _mock_answer(question, relevant)

    latency_ms = int((time.perf_counter() - start) * 1000)
    return GeneratedAnswer(text=text, citations=citations, latency_ms=latency_ms)


# --- Streaming variants -----------------------------------------------------
#
# Real token-by-token streaming, as opposed to the frontend's previous
# typewriter effect (which revealed an already-complete response client-side).
# Each of these yields text *deltas* — small pieces of the answer as they're
# produced — instead of returning one finished string.


def _mock_answer_stream(question: str, relevant: List[ScoredChunk]) -> TokenStream:
    """No real model to stream from in mock mode, so this simulates the same
    token-arrival shape (small delayed chunks) over the same canned answer
    `_mock_answer` would return — enough to exercise the real SSE pipeline
    end-to-end with zero ML/LLM dependencies, same as the rest of mock mode."""
    text = _mock_answer(question, relevant)
    words = text.split(" ")
    for i, word in enumerate(words):
        yield word if i == 0 else " " + word
        time.sleep(0.02)


def _openai_answer_stream(question: str, context: str, history: List[Message]) -> TokenStream:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("LLM_PROVIDER=openai requires the openai package — pip install openai") from exc

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    system = (
        "You are DocuMind, a RAG assistant. Answer using ONLY the numbered context passages "
        "provided. Cite claims inline with {{n}} matching the passage number, e.g. 'X causes Y {{1}}'. "
        "If the context doesn't answer the question, say so. Use the prior conversation only to "
        "understand what the user means, not as a source of facts."
    )
    messages = [{"role": "system", "content": system}]
    for m in history[-HISTORY_TURNS_FOR_REWRITE:]:
        messages.append({"role": "user" if m.role == "user" else "assistant", "content": m.content})
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})

    stream = client.chat.completions.create(model=settings.LLM_MODEL, messages=messages, stream=True)
    for event in stream:
        delta = event.choices[0].delta.content if event.choices else None
        if delta:
            yield delta


def _anthropic_answer_stream(question: str, context: str, history: List[Message]) -> TokenStream:
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "LLM_PROVIDER=anthropic requires the anthropic package — pip install anthropic"
        ) from exc

    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    system = (
        "You are DocuMind, a RAG assistant. Answer using ONLY the numbered context passages "
        "provided. Cite claims inline with {{n}} matching the passage number, e.g. 'X causes Y {{1}}'. "
        "If the context doesn't answer the question, say so. Use the prior conversation only to "
        "understand what the user means, not as a source of facts."
    )
    messages = [
        {"role": "user" if m.role == "user" else "assistant", "content": m.content}
        for m in history[-HISTORY_TURNS_FOR_REWRITE:]
    ]
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})

    with client.messages.stream(model=settings.LLM_MODEL, max_tokens=1024, system=system, messages=messages) as stream:
        for text_delta in stream.text_stream:
            yield text_delta


def generate_answer_stream(
    question: str,
    scored_chunks: List[ScoredChunk],
    documents_by_id: Dict[str, Document],
    history: List[Message] | None = None,
) -> Tuple[List[dict], TokenStream]:
    """Streaming counterpart to generate_answer(). Returns (citations, token_stream):

    - `citations` is available immediately — it comes entirely from retrieval,
      which already happened before generation starts — so callers (see
      api/chat.py's SSE endpoints) can send it to the client right away,
      before the first token of the answer exists.
    - `token_stream` is a generator yielding the answer as text deltas. Fully
      consuming it (e.g. "".join(token_stream)) reconstructs the same string
      generate_answer() would have returned as GeneratedAnswer.text.
    """
    history = history or []
    relevant, citations = _relevant_and_citations(scored_chunks, documents_by_id)

    if settings.LLM_PROVIDER == "mock" or not relevant:
        token_stream = _mock_answer_stream(question, relevant)
    else:
        context = _build_context(relevant, documents_by_id)
        if settings.LLM_PROVIDER == "openai":
            token_stream = _openai_answer_stream(question, context, history)
        elif settings.LLM_PROVIDER == "anthropic":
            token_stream = _anthropic_answer_stream(question, context, history)
        else:
            token_stream = _mock_answer_stream(question, relevant)

    return citations, token_stream
