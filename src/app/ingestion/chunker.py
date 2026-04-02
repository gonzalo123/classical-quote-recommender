"""Chunk loaded documents into retrievable snippets."""

from __future__ import annotations

import hashlib
import re

from app.domain.models import CorpusChunk, LoadedDocument

HEADING_RE = re.compile(r"^#+\s*", re.MULTILINE)


def _split_into_units(text: str) -> list[str]:
    cleaned = HEADING_RE.sub("", text)
    units = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", cleaned) if paragraph.strip()]
    return units


def chunk_document(
    document: LoadedDocument,
    chunk_size_words: int,
    overlap_words: int,
    min_chunk_words: int,
) -> list[CorpusChunk]:
    """Chunk one document using paragraph-aware word windows."""

    units = _split_into_units(document.text)
    chunks: list[CorpusChunk] = []
    current_words: list[str] = []
    chunk_number = 1

    def flush() -> None:
        nonlocal current_words, chunk_number
        if len(current_words) < min_chunk_words:
            return
        text = " ".join(current_words).strip()
        chunk_id = hashlib.sha1(
            f"{document.source_path}:{chunk_number}:{text}".encode("utf-8")
        ).hexdigest()[:16]
        chunks.append(
            CorpusChunk(
                id=chunk_id,
                text=text,
                work=document.work,
                author=document.author,
                reference=f"{document.reference_prefix} {chunk_number}",
                length=len(text.split()),
                metadata=document.metadata.copy(),
            )
        )
        overlap = current_words[-overlap_words:] if overlap_words > 0 else []
        current_words = list(overlap)
        chunk_number += 1

    for unit in units:
        words = unit.split()
        if len(current_words) + len(words) <= chunk_size_words:
            current_words.extend(words)
            continue

        if current_words:
            flush()

        while len(words) > chunk_size_words:
            current_words = words[:chunk_size_words]
            flush()
            words = words[max(chunk_size_words - overlap_words, 1) :]

        current_words.extend(words)

    if current_words:
        flush()

    return chunks


def chunk_documents(
    documents: list[LoadedDocument],
    chunk_size_words: int,
    overlap_words: int,
    min_chunk_words: int,
) -> list[CorpusChunk]:
    """Chunk all documents."""

    chunks: list[CorpusChunk] = []
    for document in documents:
        chunks.extend(
            chunk_document(
                document=document,
                chunk_size_words=chunk_size_words,
                overlap_words=overlap_words,
                min_chunk_words=min_chunk_words,
            )
        )
    return chunks
