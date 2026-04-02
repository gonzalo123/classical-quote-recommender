"""Persistence helpers for chunk metadata."""

from __future__ import annotations

import json
from pathlib import Path

from app.domain.models import CorpusChunk


def save_chunks(chunks: list[CorpusChunk], output_path: Path) -> None:
    """Persist chunks as JSONL."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(chunk.model_dump_json())
            handle.write("\n")


def load_chunks(input_path: Path) -> list[CorpusChunk]:
    """Read chunks from a JSONL file."""

    if not input_path.exists():
        return []
    chunks: list[CorpusChunk] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            chunks.append(CorpusChunk.model_validate(json.loads(line)))
    return chunks
