from app.domain.models import CorpusChunk
from app.ingestion.metadata import load_chunks, save_chunks


def test_metadata_roundtrip(tmp_path) -> None:
    output_path = tmp_path / "chunks.jsonl"
    chunks = [
        CorpusChunk(
            id="abc123",
            text="Measured speech keeps work in motion.",
            work="Sample Work",
            author="Sample Author",
            reference="fragment 1",
            length=6,
            metadata={"corpus_type": "sample"},
        )
    ]

    save_chunks(chunks, output_path)
    loaded = load_chunks(output_path)

    assert loaded == chunks
