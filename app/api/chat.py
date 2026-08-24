import json
import uuid
from datetime import datetime
from typing import Dict, Generator, List, Optional, Tuple

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.exceptions import InvalidRequestException, NotFoundException
from app.database import get_db
from app.dependencies import get_current_user
from app.models.conversation import Conversation
from app.models.document import Document
from app.models.message import Message, MessageRole
from app.schemas.chat import (
    CitationOut,
    ConversationOut,
    MessageOut,
    RegenerateRequest,
    SendMessageRequest,
)
from app.services.document_service import validate_owned_documents
from app.services.rag_service import generate_answer, generate_answer_stream, rewrite_query
from app.services.retrieval_service import retrieve_chunks

router = APIRouter(prefix="/chat", tags=["chat"])


def _get_owned_conversation(db: Session, *, owner_id: str, conversation_id: str) -> Conversation:
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.owner_id == owner_id)
        .first()
    )
    if not conv:
        raise NotFoundException("Conversation not found")
    return conv


def _retrieve_for_rag(
    db: Session,
    *,
    owner_id: str,
    question: str,
    document_ids: List[str],
    history: List[Message],
) -> Tuple[List, Dict[str, Document]]:
    """Shared by both the plain and streaming RAG paths: validates document
    ownership, retrieves chunks, and hydrates the Document rows needed for
    citation building. Returns (scored_chunks, documents_by_id) — everything
    generate_answer()/generate_answer_stream() need."""
    scoped_ids: Optional[List[str]] = document_ids or None

    # Reject up front if the caller asked for a document they don't own,
    # rather than letting retrieve_chunks silently filter it out later.
    validate_owned_documents(db, owner_id=owner_id, document_ids=document_ids)

    # Fold conversation context into the retrieval query so follow-ups like
    # "explain it with an example" retrieve chunks about what "it" refers to.
    retrieval_query = rewrite_query(question, history)
    scored_chunks = retrieve_chunks(db, query=retrieval_query, owner_id=owner_id, document_ids=scoped_ids)

    # retrieve_chunks is already owner-scoped at the DB level (it joins
    # Chunk -> Document and filters Document.owner_id == owner_id), so every
    # chunk here is guaranteed to belong to owner_id. This lookup just
    # hydrates the Document rows needed for citation building.
    doc_ids = {chunk.document_id for chunk, _ in scored_chunks}
    documents = db.query(Document).filter(Document.id.in_(doc_ids)).all()
    documents_by_id = {d.id: d for d in documents}
    return scored_chunks, documents_by_id


def _run_rag(
    db: Session,
    *,
    owner_id: str,
    question: str,
    document_ids: List[str],
    history: Optional[List[Message]] = None,
):
    history = history or []
    scored_chunks, documents_by_id = _retrieve_for_rag(
        db, owner_id=owner_id, question=question, document_ids=document_ids, history=history
    )
    return generate_answer(question, scored_chunks, documents_by_id, history=history)


def _run_rag_stream(
    db: Session,
    *,
    owner_id: str,
    question: str,
    document_ids: List[str],
    history: Optional[List[Message]] = None,
):
    """Streaming counterpart to _run_rag — same retrieval, but returns
    (citations, token_stream) instead of a finished GeneratedAnswer. See
    rag_service.generate_answer_stream for why citations are available
    before token_stream is even iterated."""
    history = history or []
    scored_chunks, documents_by_id = _retrieve_for_rag(
        db, owner_id=owner_id, question=question, document_ids=document_ids, history=history
    )
    return generate_answer_stream(question, scored_chunks, documents_by_id, history=history)


def _preceding_user_index(ordered: List[Message], cutoff: int) -> Optional[int]:
    """Index of the user turn that produced the assistant reply at position
    `cutoff`, found by scanning backward rather than assumed to be exactly
    `cutoff - 1` -- shared by _original_scope_document_ids and
    _history_before_question so both agree on which turn "the question"
    actually is. Returns None if there's no user message before `cutoff`."""
    for i in range(cutoff - 1, -1, -1):
        if ordered[i].role == MessageRole.USER:
            return i
    return None


def _original_scope_document_ids(ordered: List[Message], cutoff: int) -> List[str]:
    """The document scope a regeneration should search is the *original*
    question's scope, not whatever happens to be checked in the sidebar
    right now. The frontend used to send its currently-selected documents
    on every regenerate call, which silently reruns an old question against
    a different corpus the moment the user changes their selection —
    e.g. asking "What is A*?" with only AStar.pdf checked, later checking
    Research.pdf instead, then hitting regenerate on that old answer and
    getting it re-scoped to Research.pdf.

    Message.document_ids already stores what the *user's* question was
    scoped to when it was first asked (see models/message.py), so the fix
    is to read that back instead of trusting the request body:

        Assistant message
               ^
        preceding user message
               ^
          document_ids
               v
        regenerate using original scope

    Walks backward from `cutoff` (the position of the assistant message
    being regenerated) to find that preceding user turn. Returns []
    (unscoped -- search every document the caller owns) if there's no
    preceding user message or it wasn't scoped to anything, which matches
    how an unscoped original question behaved.
    """
    idx = _preceding_user_index(ordered, cutoff)
    if idx is None:
        return []
    doc_ids = ordered[idx].document_ids
    return json.loads(doc_ids) if doc_ids else []


def _history_before_question(ordered: List[Message], cutoff: int) -> List[Message]:
    """Everything strictly before the user question that produced the
    assistant reply at position `cutoff` -- i.e. excluding that question
    itself. send_message(_stream) builds `history` this way naturally (it
    snapshots conv.messages *before* adding the new user message -- see its
    own comment), and rag_service.py's rewrite/generate helpers are written
    assuming that convention: `question` is the current turn, `history` is
    everything that came before it.

    regenerate_message(_stream) has to rebuild the same shape from an
    already-persisted conversation instead of getting it for free, and
    getting it wrong is subtle: slicing at the *assistant* message's own
    position (`ordered[:cutoff]`) still includes the user question
    immediately before it, silently duplicating "the current question" into
    "history". That breaks rewrite_query's mock-mode fallback in particular
    -- _mock_rewrite looks for "the last user message in history" to fold
    into a short follow-up like "what byproduct does it release?", and with
    the current question sitting at the end of that history it finds itself
    instead of the real prior turn, so the rewritten query never picks up
    the earlier "A* search" context.

    Uses the same backward scan as _original_scope_document_ids so both
    agree on which turn "the question" actually is; falls back to
    `ordered[:cutoff]` (the old, occasionally-duplicating behavior) only in
    the shouldn't-happen case of no preceding user message at all.
    """
    idx = _preceding_user_index(ordered, cutoff)
    return ordered[:idx] if idx is not None else ordered[:cutoff]


def _sse(event: str, data: dict) -> str:
    """Formats one Server-Sent Event. Each event is its own JSON payload on
    a single `data:` line — SSE frames a message by the blank line after it,
    not by newlines inside `data:`, so json.dumps's default one-line output
    (no indent) is required here, not just a style choice."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


DEFAULT_TITLE = "New chat"
TITLE_MAX_LENGTH = 60


def _auto_title(text: str) -> str:
    """Derives a conversation title from its first message, the same way
    ChatGPT/Claude-style sidebars do — trims whitespace/newlines and caps
    the length so it fits a sidebar row."""
    single_line = " ".join(text.split())
    if len(single_line) <= TITLE_MAX_LENGTH:
        return single_line
    return single_line[:TITLE_MAX_LENGTH].rsplit(" ", 1)[0] + "…"


@router.post("/conversations", response_model=ConversationOut)
def create_conversation(db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    conv = Conversation(owner_id=current_user.id, title=DEFAULT_TITLE)
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return ConversationOut.from_model(conv)


@router.get("/conversations", response_model=List[ConversationOut])
def list_conversations(db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    conversations = (
        db.query(Conversation)
        .filter(Conversation.owner_id == current_user.id)
        .order_by(Conversation.updated_at.desc())
        .all()
    )
    return [ConversationOut.from_model(c) for c in conversations]


@router.get("/conversations/{conversation_id}", response_model=ConversationOut)
def get_conversation(conversation_id: str, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    conv = _get_owned_conversation(db, owner_id=current_user.id, conversation_id=conversation_id)
    return ConversationOut.from_model(conv)


@router.get("/conversations/{conversation_id}/messages", response_model=List[MessageOut])
def get_messages(conversation_id: str, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    conv = _get_owned_conversation(db, owner_id=current_user.id, conversation_id=conversation_id)
    out = []
    for msg in conv.messages:
        citations = [CitationOut(**c) for c in json.loads(msg.citations)] if msg.citations else []
        out.append(MessageOut.from_model(msg, citations))
    return out


@router.post("/conversations/{conversation_id}/messages", response_model=MessageOut)
def send_message(
    conversation_id: str,
    payload: SendMessageRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    conv = _get_owned_conversation(db, owner_id=current_user.id, conversation_id=conversation_id)

    # Snapshot prior turns *before* adding this new user message — this is
    # exactly the "oldest first, not including the current question" shape
    # rewrite_query()/generate_answer() expect (see rag_service.py). Taking
    # a plain list copy here (rather than re-reading conv.messages after
    # db.add below) avoids any risk of an autoflush during _run_rag's own
    # queries silently pulling the not-yet-answered user message into its
    # own history.
    history = list(conv.messages)

    is_first_message = len(history) == 0
    if is_first_message or conv.title == DEFAULT_TITLE:
        conv.title = _auto_title(payload.text)

    user_message = Message(
        conversation_id=conv.id,
        role=MessageRole.USER,
        content=payload.text,
        document_ids=json.dumps(payload.documentIds) if payload.documentIds else None,
    )
    db.add(user_message)

    result = _run_rag(
        db,
        owner_id=current_user.id,
        question=payload.text,
        document_ids=payload.documentIds,
        history=history,
    )

    assistant_message = Message(
        conversation_id=conv.id,
        role=MessageRole.ASSISTANT,
        content=result.text,
        citations=json.dumps(result.citations) if result.citations else None,
    )
    db.add(assistant_message)

    # Explicitly bump updated_at: onupdate only fires when a column on the
    # Conversation row itself changes during flush. Adding Message rows
    # doesn't touch Conversation at all (title only changes on the first
    # message), so without this, conversations with ongoing activity but a
    # fixed title would never resurface in GET /conversations' updated_at
    # DESC ordering.
    conv.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(assistant_message)

    citations = [CitationOut(**c) for c in result.citations]
    return MessageOut.from_model(assistant_message, citations)


@router.post("/conversations/{conversation_id}/messages/stream")
def send_message_stream(
    conversation_id: str,
    payload: SendMessageRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Real token-by-token streaming counterpart to POST .../messages: an
    SSE response instead of one JSON body returned after the full answer is
    already generated. Event sequence:

    - `start` — {messageId, conversationTitle, citations}. Sent immediately:
      citations come from retrieval, which finishes before generation
      starts, so the client can render source footnotes right away.
    - `delta` — {text} — one per chunk of the answer as it's produced.
    - `done`  — {id, text, createdAt} — sent once the full answer has been
      generated *and* persisted.
    - `error` — {message} — sent instead of `done` if generation itself
      raises partway through; nothing is persisted for this turn's reply.

    Known gap (acceptable for MVP, see event_stream()'s comment): a client
    disconnect mid-stream is NOT the same as the `error` case above and
    isn't currently handled — the partially-generated reply is silently
    lost rather than persisted or reported.
    """
    conv = _get_owned_conversation(db, owner_id=current_user.id, conversation_id=conversation_id)

    # Same "snapshot before adding the new message" reasoning as send_message.
    history = list(conv.messages)

    is_first_message = len(history) == 0
    if is_first_message or conv.title == DEFAULT_TITLE:
        conv.title = _auto_title(payload.text)

    user_message = Message(
        conversation_id=conv.id,
        role=MessageRole.USER,
        content=payload.text,
        document_ids=json.dumps(payload.documentIds) if payload.documentIds else None,
    )
    db.add(user_message)
    # Committed immediately (not batched with the assistant reply, unlike
    # the non-streaming endpoint) so the user's own message and any title
    # change are durable even if the client disconnects mid-stream.
    db.commit()

    # Retrieval — and the document-ownership check inside it — happens
    # eagerly, before the SSE stream opens, so an invalid documentId comes
    # back as a normal 4xx JSON response rather than something the client
    # has to parse out of an event stream.
    citations, token_stream = _run_rag_stream(
        db,
        owner_id=current_user.id,
        question=payload.text,
        document_ids=payload.documentIds,
        history=history,
    )

    conversation_title = conv.title
    assistant_message_id = str(uuid.uuid4())

    def event_stream() -> Generator[str, None, None]:
        # Known gap, acceptable for MVP: if the client disconnects mid-
        # stream (closed tab, dropped network, aborted fetch) rather than
        # generation itself raising, this generator never reaches the
        # persistence code below it at all -- the partial reply is lost,
        # not just unpersisted-with-a-warning.
        #
        # Why: StreamingResponse.__call__ (starlette/responses.py) races two
        # tasks in a task group -- stream_response (drives this generator)
        # and listen_for_disconnect. Whichever finishes first cancels the
        # other via task_group.cancel_scope.cancel(). On disconnect, that
        # cancels stream_response while it's mid-`async for chunk in
        # self.body_iterator`. Since this is a sync generator, body_iterator
        # is iterate_in_threadpool(event_stream()) (starlette/concurrency.py),
        # which calls next() on a worker thread via
        # anyio.to_thread.run_sync -- not cancellable by default, so that
        # one in-flight next() call runs to completion (one more `yield`
        # actually happens), but the *outer* async loop simply stops calling
        # next() again. This generator is left suspended forever at that
        # yield point, never reaching the `full_text = "".join(...)` line
        # or anything after it -- same outcome as if we'd never started.
        #
        # A tempting quick fix is wrapping the loop in try/finally (or
        # `except GeneratorExit`) to persist whatever's accumulated so far
        # when the generator is eventually closed. That's NOT reliable here:
        # `db` is closed by get_db()'s dependency teardown once
        # StreamingResponse.__call__ returns (which happens as soon as the
        # task group above exits -- promptly after cancellation, not after
        # this generator is garbage-collected), so by the time Python
        # actually calls .close() on this suspended generator and throws
        # GeneratorExit into it, `db` may well already be closed. Don't
        # reach for that pattern without also solving the session-lifetime
        # race, or it'll look like it works in casual testing and then
        # silently drop data under real network conditions.
        #
        # The right fix, when this stops being MVP-acceptable: periodically
        # checkpoint the partial answer to the Message row as deltas arrive
        # (e.g. every N tokens or every few hundred ms) instead of only
        # once at the end -- effectively a `generating` status the client
        # could also poll/reconnect against, which is what makes real
        # resumable generation possible later.
        yield _sse(
            "start",
            {
                "messageId": assistant_message_id,
                "conversationTitle": conversation_title,
                "citations": [CitationOut(**c).model_dump() for c in citations],
            },
        )

        full_text_parts: List[str] = []
        try:
            for delta in token_stream:
                full_text_parts.append(delta)
                yield _sse("delta", {"text": delta})
        except Exception:
            yield _sse("error", {"message": "The response generation failed. Please try again."})
            return

        full_text = "".join(full_text_parts)

        # `db` really is the same request-scoped session object, but its
        # identity map isn't what it was when `conv` was loaded/committed
        # above -- get_db()'s `finally: db.close()` has, in practice,
        # already run by the time this generator resumes here (FastAPI/
        # Starlette don't actually keep a sync path-operation's dependencies
        # open for a streamed response body's full duration, despite what
        # the comment here used to assume). `db.close()` doesn't make the
        # Session object unusable -- it just expunges everything from its
        # identity map and ends the transaction, auto-beginning a new one on
        # next use -- but it does mean `conv`, captured from before the
        # close, is now detached: setting conv.updated_at on it below is
        # invisible to `db`, so the commit silently persists nothing for it
        # (see regenerate_message_stream's event_stream for the sibling bug
        # where the same thing happens to an existing Message and actually
        # raises, instead of just silently no-op'ing). Re-merge it back into
        # the session first. `assistant_message` doesn't need this: it's
        # created fresh right here, never having existed in any session
        # before, so db.add() on it works regardless.
        live_conv = db.merge(conv)
        assistant_message = Message(
            id=assistant_message_id,
            conversation_id=live_conv.id,
            role=MessageRole.ASSISTANT,
            content=full_text,
            citations=json.dumps(citations) if citations else None,
        )
        db.add(assistant_message)
        # Same reasoning as send_message: new activity on this conversation
        # should resurface it in the sidebar.
        live_conv.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(assistant_message)
        created_at = assistant_message.created_at.isoformat() if assistant_message.created_at else ""

        yield _sse("done", {"id": assistant_message_id, "text": full_text, "createdAt": created_at})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            # Discourage any intermediary from buffering the whole response
            # before delivering it, which would defeat the point of streaming.
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.post("/conversations/{conversation_id}/messages/{message_id}/regenerate", response_model=MessageOut)
def regenerate_message(
    conversation_id: str,
    message_id: str,
    payload: RegenerateRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    conv = _get_owned_conversation(db, owner_id=current_user.id, conversation_id=conversation_id)
    message = db.query(Message).filter(Message.id == message_id, Message.conversation_id == conv.id).first()
    if not message:
        raise NotFoundException("Message not found")

    # Regeneration only makes sense for an assistant reply — the frontend
    # only ever wires this up from an AI message's regenerate control, but
    # the API itself doesn't otherwise enforce that. Without this check, a
    # caller could pass a user message's id and have its content silently
    # overwritten by a generated answer.
    if message.role != MessageRole.ASSISTANT:
        raise InvalidRequestException("Only assistant messages can be regenerated")

    # conv.messages is already ordered oldest-first (see Conversation.
    # messages' order_by), so slicing by position avoids any risk of
    # datetime-equality edge cases from two rows sharing a created_at.
    ordered = list(conv.messages)
    try:
        cutoff = ordered.index(message)  # position of the assistant reply being regenerated
    except ValueError:
        cutoff = len(ordered)  # shouldn't happen; fall back to full history

    # Use the original question's document scope, not payload.documentIds
    # (whatever's currently checked in the sidebar) -- see
    # _original_scope_document_ids' docstring. And use history *before* the
    # question, not before the reply -- see _history_before_question.
    result = _run_rag(
        db,
        owner_id=current_user.id,
        question=payload.userText,
        document_ids=_original_scope_document_ids(ordered, cutoff),
        history=_history_before_question(ordered, cutoff),
    )

    message.content = result.text
    message.citations = json.dumps(result.citations) if result.citations else None

    # Same reasoning as send_message: regenerating a reply is activity on
    # this conversation and should resurface it in the sidebar too.
    conv.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(message)

    citations = [CitationOut(**c) for c in result.citations]
    return MessageOut.from_model(message, citations)


@router.post("/conversations/{conversation_id}/messages/{message_id}/regenerate/stream")
def regenerate_message_stream(
    conversation_id: str,
    message_id: str,
    payload: RegenerateRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Streaming counterpart to POST .../regenerate — same event sequence as
    send_message_stream (start/delta/done/error), except the target message
    already exists: `done` means its content/citations were updated in
    place, not that a new message was created.

    Same disconnect-mid-stream gap as send_message_stream's event_stream()
    — see the detailed comment there. One difference worth knowing here:
    since this message already existed with its previous content, a
    disconnect doesn't lose data the way it does for a brand-new reply —
    `message.content` is only overwritten once the new answer is fully
    generated, so a dropped regeneration just leaves the old answer in
    place rather than losing anything. The client-side UX gap (a
    regenerate that silently never completes) is still real, just lower
    stakes than send_message_stream's case.
    """
    conv = _get_owned_conversation(db, owner_id=current_user.id, conversation_id=conversation_id)
    message = db.query(Message).filter(Message.id == message_id, Message.conversation_id == conv.id).first()
    if not message:
        raise NotFoundException("Message not found")

    # Same validation as the non-streaming regenerate endpoint — see the
    # comment there for why this matters.
    if message.role != MessageRole.ASSISTANT:
        raise InvalidRequestException("Only assistant messages can be regenerated")

    ordered = list(conv.messages)
    try:
        cutoff = ordered.index(message)  # position of the assistant reply being regenerated
    except ValueError:
        cutoff = len(ordered)

    # Use the original question's document scope, not payload.documentIds
    # (whatever's currently checked in the sidebar) -- see
    # _original_scope_document_ids' docstring. And use history *before* the
    # question, not before the reply -- see _history_before_question.
    citations, token_stream = _run_rag_stream(
        db,
        owner_id=current_user.id,
        question=payload.userText,
        document_ids=_original_scope_document_ids(ordered, cutoff),
        history=_history_before_question(ordered, cutoff),
    )

    target_message_id = message.id

    def event_stream() -> Generator[str, None, None]:
        yield _sse(
            "start",
            {
                "messageId": target_message_id,
                "citations": [CitationOut(**c).model_dump() for c in citations],
            },
        )

        full_text_parts: List[str] = []
        try:
            for delta in token_stream:
                full_text_parts.append(delta)
                yield _sse("delta", {"text": delta})
        except Exception:
            yield _sse("error", {"message": "The response generation failed. Please try again."})
            return

        full_text = "".join(full_text_parts)

        # `db` really is the same request-scoped session object -- but,
        # unlike send_message_stream (which only ever INSERTs a brand-new
        # Message from inside this generator), get_db()'s `finally: db.close()`
        # has, in practice, already run by the time this generator resumes
        # here: FastAPI/Starlette do not actually keep a sync path-operation's
        # dependencies open for the full duration of a streamed response body
        # (despite that being the assumption the original version of this
        # comment made). `db.close()` doesn't make the Session object
        # unusable -- it just expunges everything from its identity map and
        # ends the transaction, auto-beginning a new one on next use -- but
        # it does mean `message`/`conv`, captured from *before* the close,
        # are now detached. Mutating a detached instance's attributes
        # in-place doesn't error, but those changes are invisible to `db`
        # (it's no longer tracking them), and a later db.refresh(message)
        # fails outright with "not persistent within this Session". Re-merge
        # both back into the session first so they're live, tracked objects
        # again before we touch them.
        live_message = db.merge(message)
        live_conv = db.merge(conv)

        live_message.content = full_text
        live_message.citations = json.dumps(citations) if citations else None
        live_conv.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(live_message)
        created_at = live_message.created_at.isoformat() if live_message.created_at else ""

        yield _sse("done", {"id": target_message_id, "text": full_text, "createdAt": created_at})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
