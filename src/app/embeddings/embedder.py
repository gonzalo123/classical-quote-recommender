"""Embedding backends."""

from __future__ import annotations

import hashlib
import logging
from typing import Protocol

import numpy as np

from settings import (
    EMBEDDING_BACKEND,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
)

LOGGER = logging.getLogger(__name__)


class Embedder(Protocol):
    """Minimal embedder protocol."""

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts."""

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query."""


class SentenceTransformerEmbedder:
    """Sentence Transformers based embedder."""

    def __init__(self, model_name: str, device: str, batch_size: int) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)
        self._batch_size = batch_size

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        vector = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vector, dtype=np.float32)


class HashingEmbedder:
    """Deterministic local fallback used for tests and offline mode."""

    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension

    def _tokenize(self, text: str) -> list[str]:
        return [token.lower() for token in text.split() if token.strip()]

    def _embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in self._tokenize(text):
            digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
            bucket = int(digest[:8], 16) % self.dimension
            sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._embed(text) for text in texts]).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(text).astype(np.float32)


def build_embedder() -> Embedder:
    """Build the configured embedder with a graceful fallback."""

    if EMBEDDING_BACKEND == "hash":
        LOGGER.info("Using hashing embedder fallback.")
        return HashingEmbedder(dimension=EMBEDDING_DIMENSION)

    try:
        LOGGER.info("Loading Sentence Transformers model: %s", EMBEDDING_MODEL)
        return SentenceTransformerEmbedder(
            model_name=EMBEDDING_MODEL,
            device=EMBEDDING_DEVICE,
            batch_size=EMBEDDING_BATCH_SIZE,
        )
    except Exception as exc:  # pragma: no cover - depends on local runtime state
        LOGGER.warning(
            "Falling back to hashing embeddings because Sentence Transformers failed: %s",
            exc,
        )
        return HashingEmbedder(dimension=EMBEDDING_DIMENSION)
