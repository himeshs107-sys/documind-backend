"""
Database connection and SQLAlchemy configuration.

    database.py
       │
       ├── Database engine
       ├── Session factory
       ├── Base
       └── Database dependency

Runs on SQLite by default (zero setup, matches the rest of this app's
mock-first philosophy) via a synchronous engine — FastAPI runs sync
path-operation dependencies in a threadpool, so this is simple and reliable
at this app's scale.

    Later:

        FastAPI
           ↓
        SQLAlchemy
           ↓
        PostgreSQL
           ↓
        pgvector (HNSW index)

    Point DATABASE_URL at Postgres and the rest of the app follows
    automatically: models/chunk.py switches Chunk.embedding from a
    JSON-encoded Text column to a real pgvector `vector(EMBEDDING_DIM)`
    column (see IS_POSTGRES below), init_db() below builds an HNSW index
    over it, and retrieval_service.py's retrieve_chunks() issues a `<=>`
    similarity query that Postgres answers using that index instead of
    scanning every chunk in Python. See the backend README's "Migrating to
    Postgres + pgvector" for the full picture.
"""
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import settings

# --- Database engine ---
# SQLite needs check_same_thread=False since FastAPI may hand different
# requests to different threads; Postgres (or any other DB) needs no special
# connect_args at all.
is_sqlite = settings.DATABASE_URL.startswith("sqlite")
connect_args = {"check_same_thread": False} if is_sqlite else {}

engine = create_engine(settings.DATABASE_URL, connect_args=connect_args)

if is_sqlite:
    # SQLite ignores foreign key constraints unless a connection explicitly
    # turns them on — Postgres enforces them unconditionally, so without
    # this, the two backends silently disagree about what's allowed.
    # Concretely: services/document_service.py's background pipeline holds
    # a Document object across several steps (extract/chunk/embed) and
    # inserts Chunk rows referencing it at the end; if that Document gets
    # deleted mid-pipeline (see delete_document()), Postgres rejects the
    # now-orphaned Chunk inserts at commit time — caught by
    # run_processing_pipeline's existing except block, which correctly
    # no-ops once it re-fetches the (now-gone) document. Without this
    # pragma, SQLite would accept those inserts anyway, leaving orphaned
    # Chunk rows behind forever with no error and no visible symptom.
    @event.listens_for(engine, "connect")
    def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

# Whether we're on Postgres — decided once, here, from the same engine
# everything else uses. models/chunk.py reads this to pick its embedding
# column type, document_service.py reads it to decide how to serialize a
# vector before storing it, and retrieval_service.py reads it to decide
# whether search runs as a real pgvector query or the SQLite-only Python
# loop fallback. Keeping this in one place means those three files can't
# disagree about which storage format is actually in use.
IS_POSTGRES = engine.dialect.name == "postgresql"

# --- Session factory ---
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- Base ---
# Every model in app/models/ subclasses this so its table gets registered on
# Base.metadata (see init_db() below).
Base = declarative_base()


def init_db() -> None:
    """Create all tables. Called once on startup (see app/main.py).

    Order matters on Postgres: models/chunk.py declares Chunk.embedding as
    pgvector's `Vector(EMBEDDING_DIM)` type when IS_POSTGRES, and that type
    only resolves against a database that actually has the `vector`
    extension loaded. A fresh Postgres database doesn't have it loaded by
    default, so the extension has to be created *before*
    Base.metadata.create_all() ever tries to emit
    `chunks.embedding VECTOR(384)` — doing it after (the previous order
    here) works by accident on a database that already happens to have the
    extension from some earlier run, but fails `create_all()` outright on a
    genuinely fresh one. The HNSW index still has to come after
    create_all(), since it indexes a column that has to exist first:

        PostgreSQL
           |
           v
        CREATE EXTENSION IF NOT EXISTS vector   <- _ensure_pgvector_extension()
           |
           v
        CREATE TABLES                           <- create_all()
           |
           v
        CREATE HNSW INDEX                       <- _ensure_pgvector_index()

    For anything beyond local development, replace this with real migrations
    (Alembic) so schema changes are versioned and reversible instead of
    inferred from the current model definitions.
    """
    if IS_POSTGRES:
        _ensure_pgvector_extension()

    # Import models here (not at module scope) so they're registered on
    # Base.metadata before create_all runs, without creating an import
    # cycle with app/models/*.py, which import Base from this module.
    from app.models import chunk, conversation, document, evaluation_run, message, user  # noqa: F401

    Base.metadata.create_all(bind=engine)

    if IS_POSTGRES:
        _ensure_pgvector_index()


def _ensure_pgvector_extension() -> None:
    """Postgres-only: enable the pgvector extension. Must run before
    create_all() — see init_db()'s docstring for why. Idempotent
    (IF NOT EXISTS), so safe to run on every startup."""
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))


def _ensure_pgvector_index() -> None:
    """Postgres-only: build the HNSW index that makes retrieve_chunks()'s
    `<=>` query (see services/retrieval_service.py) an actual index scan
    instead of a sequential one — this is what lets vector search stay
    fast as chunks grow into the hundreds of thousands or millions, instead
    of loading every row into Python. Must run after create_all(), since it
    indexes the `chunks` table that create_all() creates.

    Idempotent (IF NOT EXISTS), so safe to run on every startup, same as
    create_all() above. This lives in startup code rather than a proper
    migration for the same reason create_all() does — see this file's
    "For anything beyond local development..." note and the README's
    "Notes / next steps": swap both for Alembic once this app needs
    versioned, reversible schema changes.
    """
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw "
                "ON chunks USING hnsw (embedding vector_cosine_ops)"
            )
        )


# --- Database dependency ---
def get_db():
    """FastAPI dependency that yields a request-scoped DB session, and
    guarantees it's closed afterward regardless of how the request ends."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
