"""
Application entry point.

    main.py
       │
       ├── Create FastAPI application
       ├── Configure CORS
       ├── Register routers
       └── Start application

Each feature area gets its own router (see app/api/):

    FastAPI
       │
       ├── /api/auth
       ├── /api/documents
       ├── /api/chat
       └── /api/evaluation
"""
import uvicorn
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api import auth, chat, documents, evaluation
from app.config import settings
from app.database import IS_POSTGRES, SessionLocal, get_db, init_db
from app.services import document_service


def create_app() -> FastAPI:
    app = FastAPI(title=settings.APP_NAME, debug=settings.DEBUG)

    configure_cors(app)
    register_routers(app)

    @app.on_event("startup")
    def on_startup() -> None:
        init_db()
        # BackgroundTasks-processed documents don't survive a crash (see
        # document_service.recover_interrupted_documents's docstring) — flag
        # any left stuck in "processing" from before this restart so the
        # user sees an actionable error instead of a spinner that never
        # moves again.
        db = SessionLocal()
        try:
            document_service.recover_interrupted_documents(db)
        finally:
            db.close()

    @app.get("/health")
    def health_check(db: Session = Depends(get_db)):
        """Liveness/readiness probe for Railway (or any orchestrator) and
        for manual debugging. Checks each dependency independently rather
        than just returning {"status": "ok"} unconditionally, so "the app
        process is up" and "the app can actually do its job" don't get
        conflated — e.g. a container that boots fine but has a wrong or
        expired DATABASE_URL would report healthy forever under the naive
        version, right up until the first real request fails.

        Returns HTTP 503 (not 200) when a check fails, on purpose: point
        Railway's healthcheck path at /health and a genuinely broken deploy
        (bad DB credentials, missing pgvector extension, ...) gets caught
        and rolled back automatically instead of going live silently.
        """
        database_status = _check_database(db)
        vector_status = _check_vector_store(db)

        # "not_configured" isn't a failure -- it's the accurate, expected
        # answer on SQLite (see database.py's IS_POSTGRES), where there's no
        # separate vector store to speak of. Only "error" should fail the
        # overall check.
        healthy = database_status == "ok" and vector_status != "error"

        return JSONResponse(
            status_code=200 if healthy else 503,
            content={
                "status": "ok" if healthy else "error",
                "app": settings.APP_NAME,
                "database": database_status,
                "vector_store": vector_status,
            },
        )

    return app


def _check_database(db: Session) -> str:
    """"ok" if a trivial query round-trips, "error" otherwise. Deliberately
    doesn't let a DB outage crash the health endpoint itself with a 500 --
    that would be far less useful than a clean "database": "error"."""
    try:
        db.execute(text("SELECT 1"))
        return "ok"
    except Exception:
        return "error"


def _check_vector_store(db: Session) -> str:
    """On SQLite there's no separate vector store (retrieval falls back to
    a Python-loop cosine search over a JSON column -- see
    retrieval_service.py), so "not_configured" is the honest answer, not a
    failure. On Postgres, actually query pg_extension rather than trusting
    IS_POSTGRES (derived from the DATABASE_URL scheme) alone -- IS_POSTGRES
    only proves we're *talking to* a Postgres server, not that the `vector`
    extension it needs is actually enabled there. init_db() enables it on
    startup (see database.py's _ensure_pgvector_extension), but this check
    exists precisely to catch the case where that step silently didn't
    happen -- an underprivileged DB role, a fresh database from outside
    init_db()'s control, etc.
    """
    if not IS_POSTGRES:
        return "not_configured"
    try:
        row = db.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")).first()
        return "ok" if row is not None else "error"
    except Exception:
        return "error"


def configure_cors(app: FastAPI) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def register_routers(app: FastAPI) -> None:
    app.include_router(auth.router, prefix=settings.API_PREFIX)
    app.include_router(documents.router, prefix=settings.API_PREFIX)
    app.include_router(chat.router, prefix=settings.API_PREFIX)
    app.include_router(evaluation.router, prefix=settings.API_PREFIX)


app = create_app()


if __name__ == "__main__":
    # Lets you start the app with `python -m app.main` as an alternative to
    # `uvicorn app.main:app --reload`.
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)
