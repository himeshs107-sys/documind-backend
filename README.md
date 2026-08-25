# DocuMind — Backend

A FastAPI RAG backend for the DocuMind app: authentication, document upload + indexing, retrieval-augmented chat with inline citations, and a lightweight evaluation endpoint.

Its API responses are shaped to match the DocuMind **frontend's `src/services/` layer exactly** — same field names, same request/response contracts — so pointing the frontend at this backend (`VITE_USE_MOCKS=false`) requires zero frontend code changes.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env             # create your local .env — see the warning below

uvicorn app.main:app --reload
# or: python -m app.main   (no --reload, but no extra CLI flags to remember)
```

The API is now at `http://localhost:8000`, with interactive docs at `http://localhost:8000/docs`. A `documind.db` SQLite file is created automatically on first run — no separate database setup needed.

> **This SQLite auto-create is local-development convenience only — it is not what should run in the cloud.** See "Deploying to the cloud" below before pushing this anywhere like Railway; a deploy with no `DATABASE_URL` set will silently boot on the same ephemeral SQLite default, which does not survive a restart/redeploy on most PaaS platforms (Railway included) since their container filesystems aren't persistent. That's a real footgun, not a hypothetical one — it's exactly what happens if you deploy this repo as-is without also setting `DATABASE_URL`.

To point the DocuMind frontend at this backend: in the frontend's `.env`, set `VITE_API_BASE_URL=http://localhost:8000/api` and `VITE_USE_MOCKS=false`.

> **Never commit `.env`.** It's already listed in `.gitignore`, and this repo intentionally does *not* ship a committed `.env` — only `.env.example` (placeholder values, safe to commit). Once you set a real `SECRET_KEY` and/or `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` in your local `.env`, those are secrets: keep them out of version control and out of anything you paste into a chat, ticket, or PR description.

## Mock mode — runs with zero ML/LLM dependencies

Just like the frontend's `VITE_USE_MOCKS` toggle, this backend defaults to mock providers so the **entire pipeline** (upload → parse → chunk → embed → retrieve → generate → cite) works immediately, with no API keys and no heavy ML libraries:

- **`EMBEDDING_PROVIDER=mock`** (default) — a deterministic hash-based pseudo-embedding (`app/services/embedding_service.py`). Same text always maps to the same vector, and texts sharing vocabulary end up closer together, which is enough to make retrieval behave sensibly for development and tests.
- **`LLM_PROVIDER=mock`** (default) — canned but citation-annotated answers (`app/services/rag_service.py`), built from whatever chunks were actually retrieved, so document-grounded answers "work" out of the box.

`requirements.txt` deliberately does **not** include `sentence-transformers`, `openai`, or `anthropic` — those are real ML/API dependencies you shouldn't need to install just to run the app in mock mode. They live in `requirements-optional.txt` instead. This is not an oversight to fix; it's the intended shape of the two setups below.

### Development setup (default — what you get with no changes)

```bash
pip install -r requirements.txt
```

```bash
# .env
EMBEDDING_PROVIDER=mock
LLM_PROVIDER=mock
```

Retrieval and generation both run, but neither is doing real semantic work — `EMBEDDING_PROVIDER=mock`'s vectors have no real semantic content (they cluster texts that share vocabulary, not texts that share meaning), and `LLM_PROVIDER=mock`'s answers are canned. Good for exercising the pipeline and running the test suite; **don't** evaluate retrieval or answer quality against this setup — see "Production setup" below before judging either.

### Production setup — real RAG generation

```bash
pip install -r requirements-optional.txt
```

Then, in `.env`, set **both** `EMBEDDING_PROVIDER` and `LLM_PROVIDER` to a real provider — each is configured independently, so pick one from each list:

**Embeddings** (pick one):
```bash
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL_NAME=text-embedding-3-small
EMBEDDING_DIM=1536
OPENAI_API_KEY=sk-...
```
```bash
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
EMBEDDING_DIM=384
# no API key needed — runs the model on this machine
```

**Answer generation** (pick one):
```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
```
```bash
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-5
ANTHROPIC_API_KEY=sk-ant-...
```

`OPENAI_API_KEY` is shared if you pick OpenAI for both — you don't need to set it twice. Mixing providers across the two settings is fine (e.g. `EMBEDDING_PROVIDER=openai` + `LLM_PROVIDER=anthropic`); they don't depend on each other.

Switching **to** or **between** real providers doesn't re-embed documents already indexed under a different provider — their stored vectors stay whatever the old provider produced, so mixing providers within one corpus silently degrades similarity scores (they're not directly comparable). Re-upload existing documents after changing `EMBEDDING_PROVIDER` (same caveat as the Postgres migration above, for the same reason: the vectors themselves, not just their storage format, differ).

### Enabling OCR (scanned/image-only PDFs)

By default, a PDF page with no real text layer (a scanned page, a photo of
a document) yields no text — pypdf can only read text that's actually
embedded in the PDF. Uploading a fully scanned PDF with `OCR_ENABLED=false`
(the default) now fails with a clear "no extractable text found" error
instead of silently indexing as an empty, unsearchable document.

To process scanned PDFs, OCR needs two system binaries that pip can't
install — **install these first**, separately from the Python packages:

```bash
# macOS
brew install tesseract poppler

# Debian/Ubuntu
sudo apt-get install tesseract-ocr poppler-utils

# Windows: see https://github.com/UB-Mannheim/tesseract/wiki (Tesseract)
#          and https://github.com/oschwartz10612/poppler-windows (Poppler)
```

Then the Python side, same as the other optional providers:

```bash
pip install -r requirements-optional.txt
```

And in `.env`:
- **`OCR_ENABLED=true`** — pypdf still runs first; only pages with fewer
  than `OCR_MIN_CHARS_PER_PAGE` characters (default 20 — catches both
  fully blank and near-blank pages) fall back to Tesseract, so mixed PDFs
  (some real text pages, some scanned) only pay the OCR cost where needed.
- **`OCR_LANGUAGE`** — Tesseract language code, default `eng`. Install the
  matching Tesseract language pack for anything else (e.g.
  `tesseract-ocr-fra` on Debian/Ubuntu for `fra`).

OCR is slower than direct text extraction — expect several seconds per
scanned page — which is part of why the processing pipeline runs as a
background task rather than blocking the upload request (see
`services/document_service.py`).

## Project structure

```
app/
  main.py            # FastAPI app: CORS, router registration, DB startup
  config.py            # Settings (env-driven), incl. the mock/real toggles
  database.py            # SQLAlchemy engine/session, init_db()
  dependencies.py          # get_db, get_current_user

  api/                # Thin routers — validate input, call services, return schemas
    auth.py             # POST /auth/register, /login, /logout, GET /auth/me
    documents.py           # POST/GET/DELETE /documents
    chat.py                   # /conversations, .../messages, .../regenerate
    evaluation.py                # POST /evaluation

  models/             # SQLAlchemy ORM: User, Document, Chunk, Conversation, Message
  schemas/            # Pydantic request/response models (camelCase — see below)

  services/           # All business logic lives here, not in routers
    auth_service.py       # register/authenticate users, issue JWTs
    document_service.py     # upload -> save -> parse -> chunk -> embed pipeline
    pdf_service.py             # PDF/DOCX/TXT -> per-page text extraction
    chunking_service.py          # page text -> overlapping chunks
    embedding_service.py           # text -> vectors (mock or sentence-transformers)
    retrieval_service.py             # cosine-similarity search over chunk embeddings
    rag_service.py                     # retrieved chunks -> answer with {{n}} citations
    evaluation_service.py                # scores an answer against expected keywords/sources

  core/
    security.py         # password hashing, JWT create/decode
    exceptions.py          # HTTPException subclasses used across the app

  utils/
    file_utils.py         # safe filenames, upload dir handling
    text_utils.py            # text cleanup, rough token counting

tests/                 # pytest suite (runs entirely on mock providers)
uploads/                 # uploaded files land here (gitignored, dir kept via .gitkeep)
```

## Matching the frontend's contract exactly

Every schema in `app/schemas/` uses **camelCase field names on purpose** (`uploadedAt`, `documentIds`, `createdAt`, ...) instead of the more Pythonic snake_case — because that's exactly what `frontend/src/services/*.js` already sends and expects. There's no translation layer in either direction:

| Frontend call (`services/*.js`) | Backend route |
|---|---|
| `authApi.login({email, password})` | `POST /api/auth/login` → `{user, token}` |
| `authApi.register({fullName, email, password})` | `POST /api/auth/register` → `{user, token}` |
| `authApi.getCurrentUser()` | `GET /api/auth/me` → `{id, name, email}` |
| `documentApi.uploadDocument(file)` | `POST /api/documents/upload` (multipart) → `{id, name, size, status, uploadedAt}` |
| `documentApi.getDocuments()` | `GET /api/documents` → `[{id, name, size, status, uploadedAt}, ...]` |
| `documentApi.getDocument(id)` | `GET /api/documents/{id}` → `{id, name, size, status, uploadedAt}` |
| `documentApi.deleteDocument(id)` | `DELETE /api/documents/{id}` → `{id, deleted}` |
| `chatApi.createConversation()` | `POST /api/chat/conversations` → `{id, title, createdAt}` |
| `chatApi.getConversations()` | `GET /api/chat/conversations` → `[{id, title, createdAt}, ...]` |
| — *(not yet called by the frontend)* | `GET /api/chat/conversations/{id}` → `{id, title, createdAt}` |
| `chatApi.getMessages(conversationId)` | `GET /api/chat/conversations/{id}/messages` → `[{id, role, text, citations, createdAt}, ...]` |
| `chatApi.sendMessage({conversationId, text, documentIds})` | `POST /api/chat/conversations/{id}/messages` → `{id, role, text, citations, createdAt}` |
| `chatApi.regenerateResponse({conversationId, messageId, userText, documentIds})` | `POST /api/chat/conversations/{id}/messages/{messageId}/regenerate` → same shape |
| — *(not yet called by the frontend)* | `POST /api/evaluation/run` → scores one question against the RAG pipeline |
| — *(not yet called by the frontend)* | `GET /api/evaluation/results` → past evaluation runs |

**Citation markers**: generated answers embed inline `{{1}}`, `{{2}}` tokens (see `rag_service.py`) — the exact convention `frontend/src/utils/citations.js`'s `parseAnswerSegments()` already parses into clickable `[n]` markers. Each citation object is `{id, source, page, snippet, url}`, matching what the frontend's `Citation`/`SourceCard`/`SourcePreview` components read directly.

**Auth**: `POST /auth/login` and `/register` return a `token` that the frontend stores in `localStorage` and sends back as `Authorization: Bearer <token>` — exactly what `frontend/src/services/api.js`'s axios interceptor already does.

## How chunking works (and its limits)

`chunking_service.chunk_pages()` turns each page's extracted text into overlapping chunks, sized by `CHUNK_SIZE`/`CHUNK_OVERLAP` (`.env`). **Those two settings are character counts, not token counts** — embedding models and LLMs reason in tokens, and ~4 characters ≈ 1 token for English as a rough rule of thumb, so the default `CHUNK_SIZE=800` is closer to ~200 tokens. Sized in characters (not run through an actual tokenizer) to keep this module dependency-free; see `config.py`'s comment on `CHUNK_SIZE` if you need exact token-budget control instead.

Chunking is structure-*aware*, not a naive fixed-offset slice: it first groups each page's text into paragraph/heading/list/code blocks (blank lines, list markers, indentation, and short unpunctuated lines are the signals used — see `chunking_service.py`'s module docstring), then packs whole blocks into each chunk, so a chunk boundary lands on a blank line instead of mid-sentence, and a heading stays attached to the section it introduces instead of ending up alone at the end of the previous chunk. A block that's too big to fit in one chunk on its own (a very long paragraph, a big code block) still falls back to character slicing for just that block, so nothing is ever dropped or produces an oversized chunk.

This is a heuristic over plain text, not a real `Document -> Heading -> Paragraph -> Semantic chunk` structural parse — `pypdf` and `python-docx` extraction (`pdf_service.py`) don't preserve font size, DOCX heading *styles*, or table layout, so there's no reliable signal downstream of extraction to build a true structure tree from. Two consequences worth knowing:

- It works best on **.docx and .txt**, where extraction preserves blank lines between paragraphs. **PDF** text extraction doesn't reliably reproduce blank lines within a page, so a PDF page often still comes back as one long block — falls back to character slicing for that block, i.e. never worse than the old behavior, just not always better.
- **Tables aren't detected or handled specially** — a table's rows are treated as ordinary lines and may be split across chunks. Recognizing tables reliably needs cell/column layout information that plain-text extraction has already discarded.

Real layout-aware structure (true heading levels, table cells) would mean extracting with something that preserves formatting — `python-docx` already exposes each paragraph's *style* (e.g. `"Heading 1"`), which isn't used yet — and passing that through instead of re-deriving it from plain text.

## How retrieval works (and its limits)

Chunk embeddings are stored as JSON-encoded float arrays in the `chunks.embedding` column, and `retrieval_service.py` computes cosine similarity in-process (pure Python/no numpy dependency required for the mock path) over whichever chunks belong to the requested `documentIds`. This is simple and dependency-free, and is fine for the corpus sizes a demo or small team would use.

**It will not scale** to a large corpus — every query does a full scan of the candidate chunks.

### Migrating to Postgres + pgvector

Vector search has two implementations, selected automatically from `DATABASE_URL` via `database.py`'s `IS_POSTGRES` flag — there's no separate feature switch:

- **SQLite (default)** — `Chunk.embedding` is a JSON-encoded `Text` column. `retrieval_service.py`'s `_search_python_loop()` loads every candidate chunk for the query and scores it in Python with cosine similarity. Fine for a handful of documents; this is the original prototype path, kept for local dev and the test suite.
- **Postgres** — `Chunk.embedding` is a real `pgvector` `vector(EMBEDDING_DIM)` column, and `_search_pgvector()` orders candidates by pgvector's `<=>` cosine-distance operator directly in SQL, so Postgres does the search instead of Python. `database.py`'s `init_db()` also builds an HNSW index (`ix_chunks_embedding_hnsw`, `vector_cosine_ops`) over the column on startup, so that query is an index scan rather than a sequential one — this is what keeps retrieval fast into the hundreds of thousands or millions of chunks, not just the tens this app ships targeting.

To switch:

1. Install the optional deps: `pip install -r requirements-optional.txt` (brings in `psycopg[binary]` and `pgvector`).
2. Point **`DATABASE_URL`** at Postgres using the `psycopg` (v3) driver, e.g. `postgresql+psycopg://user:password@localhost:5432/documind`. The Postgres server needs the `pgvector` extension available (most managed providers — RDS, Supabase, Neon, etc. — support `CREATE EXTENSION vector`, which `init_db()` runs for you).
3. Start the app. `init_db()` creates the tables (with `chunks.embedding` as a real `vector` column, since `IS_POSTGRES` is now true) and the HNSW index automatically.

Nothing else needs to change — `rag_service.py`, the API routers, and the frontend contract are all unaffected, since `retrieve_chunks()` is the only place anything touches embeddings for search. Existing SQLite-era chunks aren't migrated automatically (the storage formats differ); re-upload documents after switching, or write a one-off script that re-embeds and re-inserts them.

`init_db()` building the index at startup is itself a stopgap matching this app's "mock-first, real infra later" pattern (see `database.py`'s docstring) — for a real deployment, move both the table creation and the index into a proper Alembic migration instead, per "Notes / next steps" below.

### Deploying to the cloud (Railway + Supabase)

The production shape this repo is built toward:

```
Railway
   │
   │ FastAPI
   ▼
Supabase PostgreSQL
   │
   └── pgvector
```

Railway runs the FastAPI container (built from the `Dockerfile` in this repo); Supabase hosts the actual database — Postgres with the `pgvector` extension enabled, which is where `chunks.embedding` lives as a real `vector` column with an HNSW index (see the migration section above for the mechanism).

**Before deploying, set `DATABASE_URL` on the Railway service** to Supabase's pooled connection string (Project Settings → Database → Connection string → *Session pooler*, not the direct `:5432` connection — Railway's outbound IPs aren't static, and the pooler is also better suited to a web backend's connection pattern than a direct one):

```
DATABASE_URL=postgresql+psycopg://postgres.<project-ref>:<password>@aws-0-<region>.pooler.supabase.com:6543/postgres
```

Do this *before* the first deploy, or immediately after — not "later." Skipping it doesn't fail loudly; the app boots fine on the SQLite fallback either way (see the warning in "Getting started" above), and the difference only becomes visible the moment the container restarts and every uploaded document and conversation is gone. `GET /health`'s `vector_store` field (see below) is the fastest way to confirm which one is actually active on a running deployment: `"ok"` means real Postgres+pgvector, `"not_configured"` means it's still on the SQLite fallback.

Also set a real `SECRET_KEY` (the `.env.example` placeholder is not safe to run with) — generate one with `python -c "import secrets; print(secrets.token_urlsafe(48))"`.

## Health check

`GET /health` reports each dependency independently rather than a single unconditional `{"status": "ok"}`, so "the process is up" and "the app can actually do its job" don't get conflated:

```json
{"status": "ok", "app": "DocuMind API", "database": "ok", "vector_store": "ok"}
```

- `database`: a real `SELECT 1` round-trip against whatever `DATABASE_URL` points at.
- `vector_store`: `"not_configured"` on SQLite (expected — no separate vector store to speak of, not a failure). On Postgres, actually queries `pg_extension` rather than trusting the URL scheme alone, since that only proves the app is *talking to* a Postgres server, not that `pgvector` is genuinely enabled there.

Returns HTTP `200` when healthy, `503` when anything's actually broken — point Railway's healthcheck path at `/health` (Service Settings → Deploy → Healthcheck Path) so a broken deploy (bad DB credentials, missing `pgvector` extension, ...) gets caught and rolled back automatically instead of going live silently.

## Background processing & durability

`document_service.create_pending_document()` saves the upload and inserts the `Document` row with `status: "processing"`, then returns immediately — the parse/chunk/embed/index pipeline (`run_processing_pipeline()`) runs afterward as a FastAPI `BackgroundTask`, in the same process, after the response is already sent. The client polls `GET /documents/{id}` (or `GET /documents`) until `status` flips to `"ready"` or `"error"`, using `progress` (0-100) to render a progress bar in the meantime.

This is a deliberate choice for this project's current stage, not a placeholder — it keeps big uploads (100MB PDFs, 500-page textbooks) from tying up the request long enough to hit a client/proxy timeout, without adding infrastructure (a message broker, a separate worker process/fleet). The tradeoff to know about: **a `BackgroundTask` is not a durable job**. It's in-process state — if the server crashes or restarts while one is mid-pipeline, that job is simply gone. There's no queue to replay it from, and no other process that could pick it up.

`document_service.recover_interrupted_documents()` doesn't change that — it can't recover the lost work — but it does stop the failure from being *silent*: run once at startup (see `app/main.py`), it flags any document still stuck at `status: "processing"` from before the restart as `"error"`, so the user sees "please re-upload this" instead of a progress bar that never moves again. It only makes sense for a single-process deployment, same as `BackgroundTasks` itself — see its docstring for why running multiple app instances breaks the assumption it relies on.

**When to actually fix this**: once losing an in-flight upload on a crash/restart is a real cost (production traffic, larger files, more frequent deploys), swap `run_processing_pipeline`'s call site for a real durable job system — Celery + Redis (or RQ, Dramatiq, etc.). Nothing else in this pipeline needs to change: `create_pending_document()` already returns immediately and the client already polls for status, so the API contract stays identical: the task producer, not its shape, is what would change; `recover_interrupted_documents()` becomes unnecessary at that point, since a durable queue's jobs survive a crash and resume on their own.

## Running tests

```bash
pytest
```

Tests run against an isolated `test.db` (created/dropped automatically) and the mock embedding/LLM providers, so no network access or API keys are needed. Coverage:

- `test_auth.py` — registration, duplicate-email rejection, login, `/me`
- `test_documents.py` — upload, unsupported-type rejection, list, get, delete
- `test_chat.py` — conversation creation/retrieval, sending a message, fetching history, regenerating a reply
- `test_rag.py` — chunking behavior, mock-embedding properties, and an end-to-end check that an uploaded document's content actually gets cited in the answer
- `test_evaluation.py` — running an evaluation and confirming past runs show up in `GET /evaluation/results`

## Docker

```bash
docker build -t documind-backend .
docker run -p 8000:8000 --env-file .env documind-backend
```

Mount a volume for `uploads/` and use a real `DATABASE_URL` (e.g. Postgres) for anything beyond local development — SQLite is fine for getting started but doesn't handle concurrent writes well under real load, and won't survive a restart at all on platforms with ephemeral container filesystems (Railway included). See "Deploying to the cloud" above.

## Evaluation

`POST /evaluation/run` scores one question against the real retrieval + generation pipeline (keyword coverage in the answer, source coverage in the citations) and **persists the run** — `GET /evaluation/results` lists every past run for the current user, newest first. Neither endpoint is wired into the frontend yet; they're there for you to build a "prompt regression" workflow against (e.g. a script that runs a fixed set of questions after every change to chunking/retrieval and checks coverage hasn't regressed).

## Notes / next steps

- **Auth**: JWTs are stateless (no server-side session store), so `/auth/logout` is a no-op — add a token blocklist or move to refresh tokens if you need real server-side revocation.
- **Ownership**: every document/conversation is scoped to `owner_id`; there's no sharing/collaboration model yet (the frontend's Sidebar has a "Shared" nav item with nothing behind it yet — a natural next feature).
- **Background processing**: document indexing runs as an in-process `BackgroundTask`, not a durable job — see "Background processing & durability" above for the crash/restart tradeoff and when to move to Celery + Redis.
- **Tables**: PDF/OCR text extraction (`pdf_service.py`) flattens tables into plain lines with no cell/column structure — a table can get misread as prose or split mid-row by chunking. Not required for MVP; fixing it means layout-aware extraction (e.g. `pdfplumber`/`camelot`) plus a distinct "table" block in `chunking_service.py` — see `pdf_service.py`'s module docstring for specifics.
- **Evaluation**: `POST /evaluation/run` is a lightweight keyword/source-coverage check against a single question, not a full eval-dataset runner. Swap in a proper framework (Ragas, DeepEval, etc.) if you need statistically rigorous RAG evaluation.
- **Streaming + disconnects**: `send_message_stream`/`regenerate_message_stream` only persist the assistant reply once the full answer has streamed — if the client disconnects mid-stream (dropped network, closed tab), the partial reply is silently lost rather than saved or reported as an error; a generation failure (the `error` SSE event) already surfaces cleanly, this is specifically the disconnect case. Acceptable for MVP. See the detailed comment in `send_message_stream`'s `event_stream()` (`app/api/chat.py`) for exactly why a naive try/finally persistence patch wouldn't actually be reliable here (a `db` session-lifetime race with Starlette's `StreamingResponse` cancellation). The real fix is periodic checkpointing of the partial answer as it streams — effectively a `generating` status — which is also what would make real resumable generation possible later.
