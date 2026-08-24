"""
Central application configuration, loaded from environment variables (see
.env.example). Uses pydantic-settings so every value is validated and typed.
"""
from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- App ---
    APP_NAME: str = "DocuMind API"
    DEBUG: bool = True
    API_PREFIX: str = "/api"

    # --- CORS ---
    # Comma-separated list of allowed origins, e.g. "http://localhost:5173,https://app.example.com"
    CORS_ORIGINS: str = "http://localhost:5173"

    # --- Database ---
    DATABASE_URL: str = "sqlite:///./documind.db"

    # --- Auth / JWT ---
    SECRET_KEY: str = "change-this-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    # --- Uploads ---
    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE_MB: int = 25
    ALLOWED_EXTENSIONS: str = ".pdf,.docx,.txt"

    # --- Chunking ---
    # CHARACTER counts, not tokens. This matters because embedding models
    # and LLMs reason in tokens (roughly 4 characters ≈ 1 token for
    # English text, though this varies by tokenizer and language), not
    # characters — so CHUNK_SIZE=800 is closer to ~200 tokens, not 800.
    # Sized in characters here (rather than running an actual tokenizer)
    # to keep chunking_service.py dependency-free and fast; if you need
    # exact token-budget control (e.g. to fit a hard context-window limit
    # precisely), swap the character counts in chunking_service.py for a
    # real tokenizer (e.g. tiktoken) and convert these two settings to
    # token counts instead.
    CHUNK_SIZE: int = 800  # characters per chunk
    CHUNK_OVERLAP: int = 150  # characters of overlap between chunks

    # --- Retrieval ---
    TOP_K_RESULTS: int = 4
    MIN_SIMILARITY: float = 0.1

    # --- Embeddings ---
    # "mock"   -> deterministic hash-based vectors, zero dependencies
    #             (default) — NOT a real embedding; don't judge retrieval
    #             quality against this provider (see embedding_service.py)
    # "local"  -> a real sentence-transformers model running on this machine
    # "openai" -> OpenAI's hosted embeddings API (uses OPENAI_API_KEY below)
    #             — the production-recommended option
    EMBEDDING_PROVIDER: str = "mock"
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384  # must match the embedding provider's output size

    # --- LLM / answer generation ---
    # "mock"      -> canned, citation-annotated answers, zero dependencies (default)
    # "openai"    -> OpenAI-compatible chat completions API
    # "anthropic" -> Anthropic Messages API
    LLM_PROVIDER: str = "mock"
    LLM_MODEL: str = "gpt-4o-mini"
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""

    # --- OCR (scanned/image-only PDF pages) ---
    # Off by default — unlike EMBEDDING_PROVIDER=local, this needs system
    # binaries pip can't install (Tesseract, Poppler), not just Python
    # packages, so it's opt-in rather than "pip install and flip a flag".
    # With this off, a page with no real text layer just yields no text,
    # same as before — see requirements-optional.txt and README.md
    # "Enabling OCR" for setup.
    OCR_ENABLED: bool = False
    OCR_LANGUAGE: str = "eng"  # Tesseract language code
    # A PDF page with fewer extracted characters than this is treated as
    # "no real text layer" and sent through OCR instead (when enabled).
    # Catches near-empty pages (headers/footers only, watermark scans)
    # as well as fully blank extraction.
    OCR_MIN_CHARS_PER_PAGE: int = 20

    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    @property
    def allowed_extensions_list(self) -> List[str]:
        return [ext.strip().lower() for ext in self.ALLOWED_EXTENSIONS.split(",") if ext.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
