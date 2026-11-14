"""Утилиты чтения корпуса."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

DOC_SEP = "<|endoftext|>"


def iter_documents(path: str | Path, max_bytes: int | None = None) -> Iterator[str]:
    """Читает файл, где документы разделены строкой <|endoftext|>, по одному документу."""
    buf: list[str] = []
    seen = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            seen += len(line.encode("utf-8"))
            if line.strip() == DOC_SEP:
                if buf:
                    yield "".join(buf).strip("\n")
                    buf = []
                if max_bytes is not None and seen >= max_bytes:
                    return
            else:
                buf.append(line)
    if buf:
        yield "".join(buf).strip("\n")
