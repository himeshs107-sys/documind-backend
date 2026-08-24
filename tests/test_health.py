from unittest.mock import patch

from sqlalchemy.exc import OperationalError


def test_health_reports_ok_database_and_not_configured_vector_store_on_sqlite(client):
    # This test suite always runs against SQLite (see conftest.py), where
    # there's no separate vector store -- "not_configured" is the correct,
    # non-failing answer, not "error". See _check_vector_store's docstring.
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"
    assert body["vector_store"] == "not_configured"


def test_health_returns_503_when_database_check_fails(client):
    # Simulate a DB outage without actually tearing down the shared test
    # database other tests in this session still rely on.
    with patch("app.main.Session.execute", side_effect=OperationalError("SELECT 1", {}, Exception("connection refused"))):
        response = client.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "error"
    assert body["database"] == "error"
