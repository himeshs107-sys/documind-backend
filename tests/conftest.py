"""
Shared fixtures for the test suite: a FastAPI TestClient wired to an isolated
SQLite test database (separate from documind.db), plus a ready-made
authenticated-user fixture so most tests don't need to register/login by hand.

Runs entirely against the built-in mocks (EMBEDDING_PROVIDER=mock,
LLM_PROVIDER=mock) — no API keys or ML dependencies required.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("EMBEDDING_PROVIDER", "mock")
os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("UPLOAD_DIR", "test_uploads")
# document_service.py's background pipeline (run_processing_pipeline) opens
# its own DB session via app.database.SessionLocal rather than the get_db()
# override below — BackgroundTasks run after the request (and its
# dependency-injected session) has already completed, so it can't reuse
# that session; see that module's docstring. Left at its sqlite:///./
# documind.db default, that session would write to a different SQLite file
# than TestingSessionLocal below — a document would genuinely finish
# processing, just invisibly, in a database nothing here ever reads from.
# Pointing DATABASE_URL at the same file both session factories use keeps
# them reading and writing the same data.
os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.main import app

TEST_DATABASE_URL = os.environ["DATABASE_URL"]
# Mirrors app/database.py's is_sqlite check -- check_same_thread is a
# SQLite-only DBAPI connect kwarg; psycopg (and every other driver) rejects
# it outright, so passing it unconditionally would break `pytest
# DATABASE_URL=postgresql+psycopg://...` even though everything else about
# this file is dialect-agnostic by design.
_is_sqlite = TEST_DATABASE_URL.startswith("sqlite")
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False} if _is_sqlite else {})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = _override_get_db


@pytest.fixture(scope="session", autouse=True)
def _setup_database():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    if os.path.exists("test.db"):
        os.remove("test.db")


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture()
def auth_headers(client):
    client.post(
        "/api/auth/register",
        json={"fullName": "Test User", "email": "test@example.com", "password": "password123"},
    )
    response = client.post("/api/auth/login", json={"email": "test@example.com", "password": "password123"})
    token = response.json()["token"]
    return {"Authorization": f"Bearer {token}"}
