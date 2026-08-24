import io


def test_run_evaluation_returns_scored_result(client, auth_headers):
    content = b"A* is an informed search algorithm that uses a heuristic function."
    upload = client.post(
        "/api/documents/upload",
        headers=auth_headers,
        files={"file": ("ai_notes.txt", io.BytesIO(content), "text/plain")},
    ).json()

    response = client.post(
        "/api/evaluation/run",
        headers=auth_headers,
        json={
            "question": "What is A* search?",
            "documentIds": [upload["id"]],
            "expectedKeywords": ["heuristic"],
            "expectedSources": ["ai_notes.txt"],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["id"]
    assert body["question"] == "What is A* search?"
    assert body["numChunksRetrieved"] >= 1
    assert 0.0 <= body["keywordCoverage"] <= 1.0
    assert 0.0 <= body["sourceCoverage"] <= 1.0


def test_evaluation_requires_authentication(client):
    response = client.post("/api/evaluation/run", json={"question": "hi"})
    assert response.status_code == 401


def test_evaluation_results_lists_past_runs(client, auth_headers):
    client.post(
        "/api/evaluation/run",
        headers=auth_headers,
        json={"question": "First question", "documentIds": []},
    )
    client.post(
        "/api/evaluation/run",
        headers=auth_headers,
        json={"question": "Second question", "documentIds": []},
    )

    response = client.get("/api/evaluation/results", headers=auth_headers)
    assert response.status_code == 200
    questions = [r["question"] for r in response.json()]
    assert "First question" in questions
    assert "Second question" in questions
