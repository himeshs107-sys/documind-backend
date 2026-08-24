"""Small text-cleaning helpers used by the chunking/embedding pipeline."""
import re


def clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def approx_token_count(text: str) -> int:
    """Rough token estimate (~4 chars/token) — good enough for logging/limits
    without pulling in a real tokenizer dependency."""
    return max(1, len(text) // 4)
