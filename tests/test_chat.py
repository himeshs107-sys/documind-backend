import json


def test_create_conversation(client, auth_headers):
    response = client.post("/api/chat/conversations", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["id"]
    assert response.json()["title"] == "New chat"


def test_conversation_title_is_derived_from_first_message(client, auth_headers):
    conv = client.post("/api/chat/conversations", headers=auth_headers).json()
    client.post(
        f"/api/chat/conversations/{conv['id']}/messages",
        headers=auth_headers,
        json={"text": "What is A* search?", "documentIds": []},
    )

    response = client.get(f"/api/chat/conversations/{conv['id']}", headers=auth_headers)
    assert response.json()["title"] == "What is A* search?"


def test_conversation_title_is_truncated_when_long(client, auth_headers):
    conv = client.post("/api/chat/conversations", headers=auth_headers).json()
    long_text = "Explain in great detail how A* search differs from Dijkstra's algorithm in every possible way"
    client.post(
        f"/api/chat/conversations/{conv['id']}/messages",
        headers=auth_headers,
        json={"text": long_text, "documentIds": []},
    )

    response = client.get(f"/api/chat/conversations/{conv['id']}", headers=auth_headers)
    title = response.json()["title"]
    assert len(title) <= 61  # TITLE_MAX_LENGTH + the ellipsis character
    assert title.endswith("…")


def test_list_conversations(client, auth_headers):
    client.post("/api/chat/conversations", headers=auth_headers)
    response = client.get("/api/chat/conversations", headers=auth_headers)
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) >= 1


def test_get_single_conversation(client, auth_headers):
    conv = client.post("/api/chat/conversations", headers=auth_headers).json()
    response = client.get(f"/api/chat/conversations/{conv['id']}", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["id"] == conv["id"]


def test_get_conversation_not_found(client, auth_headers):
    response = client.get("/api/chat/conversations/does-not-exist", headers=auth_headers)
    assert response.status_code == 404


def test_send_message_returns_assistant_reply(client, auth_headers):
    conv = client.post("/api/chat/conversations", headers=auth_headers).json()

    response = client.post(
        f"/api/chat/conversations/{conv['id']}/messages",
        headers=auth_headers,
        json={"text": "What is A* search?", "documentIds": []},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["role"] == "assistant"
    assert body["text"]


def test_get_messages_includes_both_turns(client, auth_headers):
    conv = client.post("/api/chat/conversations", headers=auth_headers).json()
    client.post(
        f"/api/chat/conversations/{conv['id']}/messages",
        headers=auth_headers,
        json={"text": "Hello", "documentIds": []},
    )

    response = client.get(f"/api/chat/conversations/{conv['id']}/messages", headers=auth_headers)
    assert response.status_code == 200
    roles = [m["role"] for m in response.json()]
    assert "user" in roles
    assert "assistant" in roles


def test_regenerate_message_updates_content(client, auth_headers):
    conv = client.post("/api/chat/conversations", headers=auth_headers).json()
    first = client.post(
        f"/api/chat/conversations/{conv['id']}/messages",
        headers=auth_headers,
        json={"text": "Explain heuristics", "documentIds": []},
    ).json()

    response = client.post(
        f"/api/chat/conversations/{conv['id']}/messages/{first['id']}/regenerate",
        headers=auth_headers,
        json={"userText": "Explain heuristics", "documentIds": []},
    )
    assert response.status_code == 200
    assert response.json()["id"] == first["id"]


def test_regenerate_missing_message_is_404(client, auth_headers):
    conv = client.post("/api/chat/conversations", headers=auth_headers).json()
    response = client.post(
        f"/api/chat/conversations/{conv['id']}/messages/does-not-exist/regenerate",
        headers=auth_headers,
        json={"userText": "hi", "documentIds": []},
    )
    assert response.status_code == 404


def test_send_message_requires_authentication(client):
    response = client.post("/api/chat/conversations/some-id/messages", json={"text": "hi", "documentIds": []})
    assert response.status_code == 401


def test_conversation_reorders_after_new_message_without_title_change(client, auth_headers):
    """Regression test: a second (or later) message on a conversation must
    still bump Conversation.updated_at, even though the title only changes
    on the *first* message. Before the fix, `onupdate` only fired when a
    column on the Conversation row itself changed during flush — so an
    older conversation that keeps getting new messages would never
    resurface above a newer, untouched conversation in
    GET /conversations' `updated_at DESC` ordering.
    """
    conv_a = client.post("/api/chat/conversations", headers=auth_headers).json()
    client.post(
        f"/api/chat/conversations/{conv_a['id']}/messages",
        headers=auth_headers,
        json={"text": "First message sets the title", "documentIds": []},
    )

    # A second conversation created afterwards is naturally newer than A.
    conv_b = client.post("/api/chat/conversations", headers=auth_headers).json()

    ordering = [c["id"] for c in client.get("/api/chat/conversations", headers=auth_headers).json()]
    assert ordering.index(conv_b["id"]) < ordering.index(conv_a["id"])

    # A gets fresh activity — its title is already set, so this message
    # doesn't touch Conversation.title. It should still resurface A above B.
    client.post(
        f"/api/chat/conversations/{conv_a['id']}/messages",
        headers=auth_headers,
        json={"text": "A follow-up question, title unchanged", "documentIds": []},
    )

    ordering = [c["id"] for c in client.get("/api/chat/conversations", headers=auth_headers).json()]
    assert ordering.index(conv_a["id"]) < ordering.index(conv_b["id"])


def test_conversation_reorders_after_regenerate(client, auth_headers):
    conv_a = client.post("/api/chat/conversations", headers=auth_headers).json()
    first = client.post(
        f"/api/chat/conversations/{conv_a['id']}/messages",
        headers=auth_headers,
        json={"text": "Explain heuristics", "documentIds": []},
    ).json()

    conv_b = client.post("/api/chat/conversations", headers=auth_headers).json()
    ordering = [c["id"] for c in client.get("/api/chat/conversations", headers=auth_headers).json()]
    assert ordering.index(conv_b["id"]) < ordering.index(conv_a["id"])

    client.post(
        f"/api/chat/conversations/{conv_a['id']}/messages/{first['id']}/regenerate",
        headers=auth_headers,
        json={"userText": "Explain heuristics", "documentIds": []},
    )

    ordering = [c["id"] for c in client.get("/api/chat/conversations", headers=auth_headers).json()]
    assert ordering.index(conv_a["id"]) < ordering.index(conv_b["id"])


def test_regenerate_user_message_is_400(client, auth_headers):
    """A message id must belong to an assistant reply to be regenerated.
    Without this check, passing the *user* message's id would silently
    overwrite the user's own question with a generated answer."""
    conv = client.post("/api/chat/conversations", headers=auth_headers).json()
    client.post(
        f"/api/chat/conversations/{conv['id']}/messages",
        headers=auth_headers,
        json={"text": "Explain heuristics", "documentIds": []},
    )

    messages = client.get(f"/api/chat/conversations/{conv['id']}/messages", headers=auth_headers).json()
    user_message = next(m for m in messages if m["role"] == "user")

    response = client.post(
        f"/api/chat/conversations/{conv['id']}/messages/{user_message['id']}/regenerate",
        headers=auth_headers,
        json={"userText": "Explain heuristics", "documentIds": []},
    )
    assert response.status_code == 400


def _parse_sse(raw_text):
    """Turns a raw SSE response body into a list of (event, data) tuples,
    mirroring how the frontend's manual SSE parser splits on blank lines."""
    events = []
    for block in raw_text.strip("\n").split("\n\n"):
        if not block.strip():
            continue
        event_line, data_line = block.split("\n", 1)
        event = event_line.removeprefix("event: ")
        data = json.loads(data_line.removeprefix("data: "))
        events.append((event, data))
    return events


def test_send_message_stream_emits_start_delta_done(client, auth_headers):
    conv = client.post("/api/chat/conversations", headers=auth_headers).json()

    response = client.post(
        f"/api/chat/conversations/{conv['id']}/messages/stream",
        headers=auth_headers,
        json={"text": "What is A* search?", "documentIds": []},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(response.text)
    event_types = [e for e, _ in events]
    assert event_types[0] == "start"
    assert "delta" in event_types
    assert event_types[-1] == "done"

    start_data = events[0][1]
    assert start_data["messageId"]

    done_data = events[-1][1]
    assert done_data["id"] == start_data["messageId"]
    assert done_data["text"]

    # The deltas concatenated should reconstruct the same text reported in `done`.
    reconstructed = "".join(data["text"] for event, data in events if event == "delta")
    assert reconstructed == done_data["text"]

    # And the message is actually persisted, exactly like the non-streaming endpoint.
    messages = client.get(f"/api/chat/conversations/{conv['id']}/messages", headers=auth_headers).json()
    assert any(m["id"] == done_data["id"] and m["role"] == "assistant" for m in messages)


def test_regenerate_stream_updates_existing_message_in_place(client, auth_headers):
    conv = client.post("/api/chat/conversations", headers=auth_headers).json()
    first = client.post(
        f"/api/chat/conversations/{conv['id']}/messages/stream",
        headers=auth_headers,
        json={"text": "Explain heuristics", "documentIds": []},
    )
    first_message_id = _parse_sse(first.text)[-1][1]["id"]

    response = client.post(
        f"/api/chat/conversations/{conv['id']}/messages/{first_message_id}/regenerate/stream",
        headers=auth_headers,
        json={"userText": "Explain heuristics", "documentIds": []},
    )
    assert response.status_code == 200
    events = _parse_sse(response.text)
    assert events[0][0] == "start"
    assert events[-1][0] == "done"
    assert events[-1][1]["id"] == first_message_id  # same message, updated in place — not a new one


def test_regenerate_stream_rejects_user_message(client, auth_headers):
    conv = client.post("/api/chat/conversations", headers=auth_headers).json()
    client.post(
        f"/api/chat/conversations/{conv['id']}/messages/stream",
        headers=auth_headers,
        json={"text": "Explain heuristics", "documentIds": []},
    )
    messages = client.get(f"/api/chat/conversations/{conv['id']}/messages", headers=auth_headers).json()
    user_message = next(m for m in messages if m["role"] == "user")

    response = client.post(
        f"/api/chat/conversations/{conv['id']}/messages/{user_message['id']}/regenerate/stream",
        headers=auth_headers,
        json={"userText": "Explain heuristics", "documentIds": []},
    )
    assert response.status_code == 400


def test_send_message_to_unknown_conversation_is_404(client, auth_headers):
    response = client.post(
        "/api/chat/conversations/does-not-exist/messages",
        headers=auth_headers,
        json={"text": "hi", "documentIds": []},
    )
    assert response.status_code == 404
