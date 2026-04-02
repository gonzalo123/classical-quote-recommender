from app.domain.models import LoadedDocument
from app.ingestion.chunker import chunk_document


def test_chunk_document_produces_stable_chunks() -> None:
    document = LoadedDocument(
        source_path="sample.md",
        text=(
            "One measured reply keeps the team moving through disagreement.\n\n"
            "When voices harden, shared purpose matters more than private pride.\n\n"
            "A practical compromise often saves the voyage."
        ),
        author="Sample",
        work="Sample Work",
        reference_prefix="fragment",
    )

    chunks = chunk_document(document, chunk_size_words=12, overlap_words=3, min_chunk_words=6)

    assert len(chunks) >= 2
    assert chunks[0].reference == "fragment 1"
    assert chunks[0].id
    assert chunks[0].length >= 6
