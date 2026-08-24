import io
import math

from app.services import chunking_service, embedding_service


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def test_chunking_splits_long_text_with_overlap():
    text = "sentence. " * 400  # long enough to require multiple chunks
    chunks = chunking_service.chunk_pages([(1, text)], chunk_size=200, overlap=50)
    assert len(chunks) > 1
    assert all(c.page_number == 1 for c in chunks)
    assert chunks[0].content[-20:] in text


def test_chunking_respects_page_boundaries():
    pages = [(1, "First page content here."), (2, "Second page content here.")]
    chunks = chunking_service.chunk_pages(pages, chunk_size=1000, overlap=0)
    assert [c.page_number for c in chunks] == [1, 2]


def test_mock_embeddings_are_deterministic():
    vec_a = embedding_service.embed_query("heuristic search algorithm")
    vec_b = embedding_service.embed_query("heuristic search algorithm")
    assert vec_a == vec_b


def test_mock_embeddings_are_normalized():
    vec = embedding_service.embed_query("heuristic search algorithm")
    norm = math.sqrt(sum(v * v for v in vec))
    assert abs(norm - 1.0) < 1e-6 or norm == 0.0


def test_mock_embeddings_ignore_punctuation():
    vec_a = embedding_service.embed_query("What is A* search?")
    vec_b = embedding_service.embed_query("what is a search")
    # Shared vocabulary (once punctuation is stripped) should dominate, so
    # these two should be far closer than two unrelated sentences.
    unrelated = embedding_service.embed_query("chocolate chip cookie recipe")
    assert _cosine(vec_a, vec_b) > _cosine(vec_a, unrelated)


def test_similar_texts_embed_closer_than_unrelated_text():
    a = embedding_service.embed_query("heuristic search algorithm A* pathfinding")
    b = embedding_service.embed_query("A* pathfinding uses a heuristic function")
    c = embedding_service.embed_query("chocolate chip cookie recipe ingredients")

    assert _cosine(a, b) > _cosine(a, c)


def test_end_to_end_rag_cites_the_uploaded_document(client, auth_headers):
    content = (
        b"A* is an informed search algorithm. "
        b"It uses a heuristic function to estimate the cost to the goal, "
        b"guiding the search toward the most promising paths first."
    )
    upload = client.post(
        "/api/documents/upload",
        headers=auth_headers,
        files={"file": ("ai_notes.txt", io.BytesIO(content), "text/plain")},
    ).json()
    # Upload returns before background processing runs, so the response body
    # itself still says "processing" — but TestClient executes BackgroundTasks
    # synchronously within the same call, so by the time we poll here the
    # pipeline has already finished.
    ready = client.get(f"/api/documents/{upload['id']}", headers=auth_headers).json()
    assert ready["status"] == "ready"

    conv = client.post("/api/chat/conversations", headers=auth_headers).json()
    response = client.post(
        f"/api/chat/conversations/{conv['id']}/messages",
        headers=auth_headers,
        json={"text": "What is A* search?", "documentIds": [upload["id"]]},
    )
    body = response.json()

    assert body["citations"], "expected at least one citation from the uploaded document"
    assert body["citations"][0]["source"] == "ai_notes.txt"
    # The backend intentionally leaves {{n}} markers IN the text — the
    # frontend's utils/citations.js parses them into clickable [n] markers.
    assert "{{1}}" in body["text"], "answer should embed a {{1}} citation marker for the frontend to parse"


def test_rag_with_no_matching_documents_returns_no_citations(client, auth_headers):
    # "No matching documents" means retrieval legitimately finds nothing
    # relevant, not an invalid document ID -- an unowned/nonexistent ID in
    # documentIds is rejected up front with 404 by validate_owned_documents
    # (see its docstring), so it never reaches retrieval at all.
    #
    # documentIds=[] (meaning "search all of this user's documents") isn't
    # reliable here either: auth_headers logs into the same shared user for
    # every test in this session-scoped test database (see conftest.py), so
    # by the time this test runs, that user may already own documents
    # uploaded by earlier tests -- including ones genuinely about A* search.
    # Instead, scope explicitly to a real, freshly-uploaded document whose
    # content has nothing in common with the query, so retrieval's
    # MIN_SIMILARITY cutoff legitimately excludes it regardless of test
    # order or what else this user owns.
    unrelated = client.post(
        "/api/documents/upload",
        headers=auth_headers,
        files={
            "file": (
                "photosynthesis_unrelated.txt",
                io.BytesIO(b"Photosynthesis converts sunlight into chemical energy stored in glucose."),
                "text/plain",
            )
        },
    ).json()
    ready = client.get(f"/api/documents/{unrelated['id']}", headers=auth_headers).json()
    assert ready["status"] == "ready"

    conv = client.post("/api/chat/conversations", headers=auth_headers).json()
    response = client.post(
        f"/api/chat/conversations/{conv['id']}/messages",
        headers=auth_headers,
        json={"text": "What is A* search?", "documentIds": [unrelated["id"]]},
    )
    body = response.json()
    assert body["citations"] == []


def test_txt_citation_has_no_page_number(client, auth_headers):
    # .txt (and .docx) have no reliable page concept — extract_pages()
    # reports page_number=None for them rather than a faked page 1 (see
    # pdf_service.py). This exercises that all the way through retrieval
    # and citation-building, not just the extraction unit in
    # test_pdf_service.py — a regression anywhere in that chain (chunking,
    # the Chunk model, _build_citation) should fail this.
    content = (
        b"A* is an informed search algorithm. "
        b"It uses a heuristic function to estimate the cost to the goal."
    )
    upload = client.post(
        "/api/documents/upload",
        headers=auth_headers,
        files={"file": ("plain_notes.txt", io.BytesIO(content), "text/plain")},
    ).json()
    ready = client.get(f"/api/documents/{upload['id']}", headers=auth_headers).json()
    assert ready["status"] == "ready"

    conv = client.post("/api/chat/conversations", headers=auth_headers).json()
    response = client.post(
        f"/api/chat/conversations/{conv['id']}/messages",
        headers=auth_headers,
        json={"text": "What is A* search?", "documentIds": [upload["id"]]},
    )
    body = response.json()

    assert body["citations"], "expected at least one citation from the uploaded document"
    assert body["citations"][0]["page"] is None
    # #page=N is meaningless for a .txt file (see _build_citation) — the
    # file URL should be bare, with no fragment appended.
    assert "#page=" not in body["citations"][0]["url"]


def test_followup_question_uses_conversation_history_for_retrieval(client, auth_headers):
    # Two documents on unrelated topics. This specific bare follow-up —
    # "What byproduct does it release?" — actually scores *closer* to the
    # photosynthesis document than the A* one on its own (both chunks
    # mention "byproduct"/"release"-adjacent vocabulary), so this only
    # lands on the right (A*) document if send_message actually threads
    # conversation history into _run_rag -> rewrite_query, which folds the
    # prior "What is A* search?" question in before embedding the query.
    # A regression back to history=[] would make this test fail, not just
    # weakly pass — see the bare-vs-rewritten similarity check this test
    # was derived from.
    astar_doc = client.post(
        "/api/documents/upload",
        headers=auth_headers,
        files={
            "file": (
                "astar.txt",
                io.BytesIO(
                    b"A* is an informed search algorithm. It uses a heuristic function "
                    b"to estimate the cost to the goal, guiding the search toward the "
                    b"most promising paths first."
                ),
                "text/plain",
            )
        },
    ).json()
    photosynthesis_doc = client.post(
        "/api/documents/upload",
        headers=auth_headers,
        files={
            "file": (
                "photosynthesis.txt",
                io.BytesIO(
                    b"Photosynthesis is the process plants use to convert light energy "
                    b"into chemical energy stored in glucose, releasing oxygen as a "
                    b"byproduct through reactions in the chloroplast."
                ),
                "text/plain",
            )
        },
    ).json()

    for doc in (astar_doc, photosynthesis_doc):
        status = client.get(f"/api/documents/{doc['id']}", headers=auth_headers).json()
        assert status["status"] == "ready"

    conv = client.post("/api/chat/conversations", headers=auth_headers).json()
    both_doc_ids = [astar_doc["id"], photosynthesis_doc["id"]]

    first = client.post(
        f"/api/chat/conversations/{conv['id']}/messages",
        headers=auth_headers,
        json={"text": "What is A* search?", "documentIds": both_doc_ids},
    ).json()
    assert first["citations"], "expected the first answer to cite the A* document"
    assert first["citations"][0]["source"] == "astar.txt"

    followup = client.post(
        f"/api/chat/conversations/{conv['id']}/messages",
        headers=auth_headers,
        json={"text": "What byproduct does it release?", "documentIds": both_doc_ids},
    ).json()
    assert followup["citations"], "follow-up should still retrieve relevant chunks using conversation context"
    assert followup["citations"][0]["source"] == "astar.txt", (
        "without history, 'What byproduct does it release?' actually scores "
        "closer to photosynthesis.txt than astar.txt — this only passes if "
        "conversation history is folded into the retrieval query"
    )


def test_regenerate_uses_history_before_the_regenerated_message(client, auth_headers):
    # Same idea as the send_message test above, but for the regenerate
    # endpoint: history must be everything *before* the message being
    # regenerated, not the full conversation and not empty.
    astar_doc = client.post(
        "/api/documents/upload",
        headers=auth_headers,
        files={
            "file": (
                "astar.txt",
                io.BytesIO(
                    b"A* is an informed search algorithm. It uses a heuristic function "
                    b"to estimate the cost to the goal, guiding the search toward the "
                    b"most promising paths first."
                ),
                "text/plain",
            )
        },
    ).json()
    photosynthesis_doc = client.post(
        "/api/documents/upload",
        headers=auth_headers,
        files={
            "file": (
                "photosynthesis.txt",
                io.BytesIO(
                    b"Photosynthesis is the process plants use to convert light energy "
                    b"into chemical energy stored in glucose, releasing oxygen as a "
                    b"byproduct through reactions in the chloroplast."
                ),
                "text/plain",
            )
        },
    ).json()
    for doc in (astar_doc, photosynthesis_doc):
        status = client.get(f"/api/documents/{doc['id']}", headers=auth_headers).json()
        assert status["status"] == "ready"

    conv = client.post("/api/chat/conversations", headers=auth_headers).json()
    both_doc_ids = [astar_doc["id"], photosynthesis_doc["id"]]

    client.post(
        f"/api/chat/conversations/{conv['id']}/messages",
        headers=auth_headers,
        json={"text": "What is A* search?", "documentIds": both_doc_ids},
    )
    followup = client.post(
        f"/api/chat/conversations/{conv['id']}/messages",
        headers=auth_headers,
        json={"text": "What byproduct does it release?", "documentIds": both_doc_ids},
    ).json()

    regenerated = client.post(
        f"/api/chat/conversations/{conv['id']}/messages/{followup['id']}/regenerate",
        headers=auth_headers,
        json={"userText": "What byproduct does it release?", "documentIds": both_doc_ids},
    ).json()
    assert regenerated["citations"], "regeneration should still resolve 'it' using the conversation before this message"
    assert regenerated["citations"][0]["source"] == "astar.txt"
