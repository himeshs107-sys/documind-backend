"""Filesystem helpers for uploaded documents."""
import os
import re


def ensure_upload_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def safe_filename(filename: str) -> str:
    """Strips path separators and anything that isn't alnum/dot/dash/underscore,
    so a filename can't be used to escape the upload directory."""
    name = os.path.basename(filename)
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name[:200] or "file"


def human_file_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"
