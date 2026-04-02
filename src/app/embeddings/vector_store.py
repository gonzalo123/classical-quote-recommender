"""Local vector store backends."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class NumpyVectorStore:
    """Simple local cosine-similarity store."""

    def __init__(self, ids: list[str] | None = None, vectors: np.ndarray | None = None) -> None:
        self.ids = ids or []
        self.vectors = (
            np.asarray(vectors, dtype=np.float32)
            if vectors is not None
            else np.zeros((0, 0), dtype=np.float32)
        )

    def build(self, ids: list[str], vectors: np.ndarray) -> None:
        self.ids = list(ids)
        self.vectors = np.asarray(vectors, dtype=np.float32)

    def save(self, index_path: Path) -> None:
        index_path.mkdir(parents=True, exist_ok=True)
        np.save(index_path / "vectors.npy", self.vectors)
        (index_path / "ids.json").write_text(json.dumps(self.ids), encoding="utf-8")

    @classmethod
    def load(cls, index_path: Path) -> "NumpyVectorStore":
        vectors_path = index_path / "vectors.npy"
        ids_path = index_path / "ids.json"
        if not vectors_path.exists() or not ids_path.exists():
            raise FileNotFoundError(f"Index files not found in {index_path}")
        vectors = np.load(vectors_path)
        ids = json.loads(ids_path.read_text(encoding="utf-8"))
        return cls(ids=ids, vectors=vectors)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if self.vectors.size == 0 or not self.ids:
            return []
        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim > 1:
            query = query[0]
        if self.vectors.ndim != 2 or self.vectors.shape[1] != query.shape[0]:
            raise ValueError(
                "Embedding dimension mismatch between the persisted index and the active "
                "embedder. Re-run corpus ingestion to rebuild the index."
            )
        scores = self.vectors @ query
        indices = np.argsort(scores)[::-1][:top_k]
        return [(self.ids[index], float(scores[index])) for index in indices]
