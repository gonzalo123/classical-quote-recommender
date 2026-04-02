"""Semantic retrieval on top of the vector store."""

from __future__ import annotations

from app.domain.models import CorpusChunk
from app.embeddings.embedder import Embedder
from app.embeddings.vector_store import NumpyVectorStore


class SemanticRetriever:
    """Nearest-neighbor retriever."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: NumpyVectorStore,
        chunks_by_id: dict[str, CorpusChunk],
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.chunks_by_id = chunks_by_id

    def search(self, query: str, top_k: int) -> list[tuple[CorpusChunk, float]]:
        query_vector = self.embedder.embed_query(query)
        results = self.vector_store.search(query_vector=query_vector, top_k=top_k)
        return [
            (self.chunks_by_id[chunk_id], score)
            for chunk_id, score in results
            if chunk_id in self.chunks_by_id
        ]
