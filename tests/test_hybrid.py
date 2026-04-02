from app.domain.models import CorpusChunk, InputAnalysis
from app.embeddings.embedder import HashingEmbedder
from app.embeddings.vector_store import NumpyVectorStore
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.lexical import LexicalRetriever
from app.retrieval.semantic import SemanticRetriever


def test_hybrid_retrieval_prefers_middle_ground_quote() -> None:
    chunks = [
        CorpusChunk(
            id="q1",
            text="Two captains reached the harbor when each yielded a little pride and found middle ground for the common voyage.",
            work="Sample Epic",
            author="Sample Corpus",
            reference="fragment 1",
            length=19,
            metadata={},
        ),
        CorpusChunk(
            id="q2",
            text="Pride builds a wall faster than grief can climb it.",
            work="Sample Tragedy",
            author="Sample Corpus",
            reference="fragment 2",
            length=10,
            metadata={},
        ),
        CorpusChunk(
            id="q3",
            text="The calm mind orders conflict so action can follow.",
            work="Sample Stoa",
            author="Sample Corpus",
            reference="fragment 3",
            length=10,
            metadata={},
        ),
    ]
    embedder = HashingEmbedder()
    store = NumpyVectorStore()
    store.build([chunk.id for chunk in chunks], embedder.embed_texts([chunk.text for chunk in chunks]))
    chunks_by_id = {chunk.id: chunk for chunk in chunks}

    retriever = HybridRetriever(
        semantic_retriever=SemanticRetriever(embedder, store, chunks_by_id),
        lexical_retriever=LexicalRetriever(),
        chunks_by_id=chunks_by_id,
    )

    analysis = InputAnalysis(
        summary="A polite disagreement that looks for a compromise.",
        main_theme="negotiation",
        secondary_themes=["teamwork", "feedback"],
        tone="conciliatory",
        intent="negotiate",
        dominant_emotion="controlled tension",
        recommended_quote_type="measured and diplomatic",
    )

    results = retriever.retrieve(
        text="I do not fully agree, but I think we can find a middle ground and move forward.",
        analysis=analysis,
        top_k=3,
    )

    assert results[0].chunk.id == "q1"
