def test_register_creates_user_and_returns_token(client):
    response = client.post(
        "/api/auth/register",
        json={"fullName": "Ada Lovelace", "email": "ada@example.com", "password": "supersecret"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["user"]["email"] == "ada@example.com"
    assert body["user"]["name"] == "Ada Lovelace"
    assert body["token"]


def test_register_rejects_duplicate_email(client):
    payload = {"fullName": "Grace Hopper", "email": "grace@example.com", "password": "supersecret"}
    first = client.post("/api/auth/register", json=payload)
    assert first.status_code == 200

    second = client.post("/api/auth/register", json=payload)
    assert second.status_code == 409


def test_register_rejects_short_password(client):
    response = client.post(
        "/api/auth/register",
        json={"fullName": "Short Pass", "email": "short@example.com", "password": "abc"},
    )
    assert response.status_code == 422


def test_login_with_correct_credentials(client):
    client.post(
        "/api/auth/register",
        json={"fullName": "Alan Turing", "email": "alan@example.com", "password": "supersecret"},
    )
    response = client.post("/api/auth/login", json={"email": "alan@example.com", "password": "supersecret"})
    assert response.status_code == 200
    assert response.json()["token"]


def test_login_with_wrong_password_is_rejected(client):
    client.post(
        "/api/auth/register",
        json={"fullName": "Barbara Liskov", "email": "barbara@example.com", "password": "supersecret"},
    )
    response = client.post("/api/auth/login", json={"email": "barbara@example.com", "password": "wrong"})
    assert response.status_code == 401


def test_me_requires_authentication(client):
    response = client.get("/api/auth/me")
    assert response.status_code == 401


def test_me_returns_current_user(client, auth_headers):
    response = client.get("/api/auth/me", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["email"] == "test@example.com"
