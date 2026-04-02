"""Main orchestration service for the quote recommender."""

from __future__ import annotations

import logging
from pathlib import Path

from app.agents.analyzer import InputAnalyzer
from app.agents.selector import QuoteSelector
from app.domain.models import CorpusChunk, QuoteOption
from app.embeddings.embedder import build_embedder
from app.embeddings.vector_store import NumpyVectorStore
from app.ingestion.chunker import chunk_documents
from app.ingestion.loader import load_corpus
from app.ingestion.metadata import load_chunks, save_chunks
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.lexical import LexicalRetriever
from app.retrieval.reranker import RuleBasedReranker
from app.retrieval.semantic import SemanticRetriever
from app.schemas import AnalyzeResponse, IngestResponse, QuoteResponse
from app.services.localization_service import LocalizationService
from settings import (
    CHUNK_OVERLAP_WORDS,
    CHUNK_SIZE_WORDS,
    DATA_INDEX_PATH,
    DATA_PROCESSED_PATH,
    DATA_RAW_PATH,
    MIN_CHUNK_WORDS,
    RERANK_CANDIDATE_COUNT,
    VECTOR_BACKEND,
    EMBEDDING_BACKEND,
)

LOGGER = logging.getLogger(__name__)


class QuoteService:
    """High-level service used by the API and the CLI."""

    def __init__(self) -> None:
        self.analyzer = InputAnalyzer()
        self.selector = QuoteSelector()
        self.localization_service = LocalizationService()
        self.reranker = RuleBasedReranker()
        self._embedder = None
        self._vector_store: NumpyVectorStore | None = None
        self._chunks_by_id: dict[str, CorpusChunk] = {}

    def _get_embedder(self):
        if self._embedder is None:
            self._embedder = build_embedder()
        return self._embedder

    def _load_index_if_needed(self) -> None:
        if self._vector_store is not None and self._chunks_by_id:
            return
        chunks = load_chunks(DATA_PROCESSED_PATH)
        if not chunks:
            raise FileNotFoundError(
                "No processed corpus found. Run ingestion first with POST /ingest or the CLI."
            )
        self._chunks_by_id = {chunk.id: chunk for chunk in chunks}
        self._vector_store = NumpyVectorStore.load(DATA_INDEX_PATH)

    def ingest(self, input_path: Path | None = None) -> IngestResponse:
        """Build the local index from the raw corpus."""

        source_path = input_path or DATA_RAW_PATH
        documents = load_corpus(source_path)
        chunks = chunk_documents(
            documents=documents,
            chunk_size_words=CHUNK_SIZE_WORDS,
            overlap_words=CHUNK_OVERLAP_WORDS,
            min_chunk_words=MIN_CHUNK_WORDS,
        )
        if not chunks:
            raise ValueError(f"No chunks were produced from {source_path}")

        save_chunks(chunks, DATA_PROCESSED_PATH)
        embedder = self._get_embedder()
        vectors = embedder.embed_texts([chunk.text for chunk in chunks])
        vector_store = NumpyVectorStore()
        vector_store.build(ids=[chunk.id for chunk in chunks], vectors=vectors)
        vector_store.save(DATA_INDEX_PATH)

        self._vector_store = vector_store
        self._chunks_by_id = {chunk.id: chunk for chunk in chunks}
        LOGGER.info("Indexed %s documents into %s chunks.", len(documents), len(chunks))

        return IngestResponse(
            input_path=str(source_path),
            documents=len(documents),
            chunks=len(chunks),
            embedding_backend=EMBEDDING_BACKEND,
            vector_backend=VECTOR_BACKEND,
            metadata_path=str(DATA_PROCESSED_PATH),
            index_path=str(DATA_INDEX_PATH),
        )

    def _quote_source_language(self, chunk: CorpusChunk) -> str:
        return str(chunk.metadata.get("language", "en")).lower()

    def analyze(self, text: str) -> AnalyzeResponse:
        """Analyze input text."""

        analyzed = self.analyzer.analyze(text)
        return AnalyzeResponse(
            detected_language=analyzed.detected_language,
            input_analysis=analyzed.input_analysis,
        )

    def quote(self, text: str) -> QuoteResponse:
        """Return the best rhetorical quote plus alternatives."""

        self._load_index_if_needed()
        if self._vector_store is None:
            raise FileNotFoundError("Vector store is not available.")

        analyzed = self.analyzer.analyze(text)
        detected_language = analyzed.detected_language
        analysis = analyzed.input_analysis
        semantic = SemanticRetriever(
            embedder=self._get_embedder(),
            vector_store=self._vector_store,
            chunks_by_id=self._chunks_by_id,
        )
        lexical = LexicalRetriever()
        hybrid = HybridRetriever(
            semantic_retriever=semantic,
            lexical_retriever=lexical,
            chunks_by_id=self._chunks_by_id,
        )
        candidates = hybrid.retrieve(
            text=text,
            analysis=analysis,
            top_k=RERANK_CANDIDATE_COUNT,
        )
        reranked = self.reranker.rerank(text, analysis, candidates)
        selection = self.selector.select(text, analysis, reranked, detected_language)

        quote_lookup = {candidate.chunk.id: candidate for candidate in reranked}
        ranked_options: list[QuoteOption] = []
        for choice in selection.ranked_quotes:
            candidate = quote_lookup.get(choice.quote_id)
            if candidate is None:
                continue
            quote_text = candidate.quote_text or candidate.chunk.text
            source_language = self._quote_source_language(candidate.chunk)
            translated_text = self.localization_service.translate_quote(
                quote_text,
                source_language=source_language,
                target_language=detected_language,
            )
            ranked_options.append(
                QuoteOption(
                    quote_id=candidate.chunk.id,
                    text=quote_text,
                    translated_text=translated_text,
                    author=candidate.chunk.author,
                    work=candidate.chunk.work,
                    reference=candidate.chunk.reference,
                    score=choice.score,
                    why_it_fits=choice.why_it_fits,
                )
            )

        if not ranked_options:
            raise ValueError("No quote candidates could be selected.")

        recommended = next(
            (
                option
                for option in ranked_options
                if option.quote_id == selection.recommended_quote_id
            ),
            ranked_options[0],
        )
        alternatives = [option for option in ranked_options if option.quote_id != recommended.quote_id]

        return QuoteResponse(
            detected_language=detected_language,
            input_analysis=analysis,
            recommended_quote=recommended,
            alternatives=alternatives,
        )

    def get_quote(self, quote_id: str) -> QuoteOption | None:
        """Return a single quote by id from the indexed corpus."""

        self._load_index_if_needed()
        chunk = self._chunks_by_id.get(quote_id)
        if chunk is None:
            return None
        return QuoteOption(
            quote_id=chunk.id,
            text=chunk.text,
            translated_text=chunk.text,
            author=chunk.author,
            work=chunk.work,
            reference=chunk.reference,
            score=0.0,
            why_it_fits="Direct lookup from the indexed corpus.",
        )
