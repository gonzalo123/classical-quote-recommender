from app.domain.models import CorpusChunk, InputAnalysis, RetrievedCandidate
from app.retrieval.quote_extractor import extract_compact_quote
from app.retrieval.reranker import RuleBasedReranker


def test_extract_compact_quote_returns_complete_sentence_window() -> None:
    analysis = InputAnalysis(
        summary="A message that asks for a measured compromise.",
        main_theme="negotiation",
        secondary_themes=["agreement"],
        tone="conciliatory",
        intent="negotiate",
        dominant_emotion="calm",
        recommended_quote_type="measured and concise",
    )

    extracted = extract_compact_quote(
        (
            "Many voices rose in anger across the hall. "
            "A measured answer kept both captains moving toward the same harbor. "
            "So the council ended without another wound."
        ),
        "We can still find a middle ground and move forward.",
        analysis,
    )

    assert extracted.text == "A measured answer kept both captains moving toward the same harbor."
    assert extracted.word_count <= 32
    assert extracted.sentence_count == 1
    assert extracted.has_clear_boundaries is True


def test_reranker_derives_compact_complete_quote_windows() -> None:
    analysis = InputAnalysis(
        summary="A short disagreement looking for compromise.",
        main_theme="negotiation",
        secondary_themes=["teamwork"],
        tone="conciliatory",
        intent="negotiate",
        dominant_emotion="restraint",
        recommended_quote_type="brief diplomatic quote",
    )
    reranker = RuleBasedReranker()
    candidates = [
        RetrievedCandidate(
            chunk=CorpusChunk(
                id="q1",
                text=(
                    "The argument spread through the camp without end and every grievance "
                    "was repeated until nightfall. A measured answer kept both captains "
                    "moving toward the same harbor. So the council ended without another wound."
                ),
                work="Sample Epic",
                author="Sample Author",
                reference="fragment 1",
                length=33,
                metadata={},
            ),
            hybrid_score=0.85,
        ),
        RetrievedCandidate(
            chunk=CorpusChunk(
                id="q2",
                text=(
                    "Many speeches followed and none of them yielded. The sea was dark."
                ),
                work="Sample Epic",
                author="Sample Author",
                reference="fragment 2",
                length=12,
                metadata={},
            ),
            hybrid_score=0.84,
        ),
    ]

    reranked = reranker.rerank(
        "I do not fully agree, but I think we can still find middle ground.",
        analysis,
        candidates,
    )

    q1 = next(candidate for candidate in reranked if candidate.chunk.id == "q1")

    assert q1.quote_sentence_count == 1
    assert q1.quote_has_clear_boundaries is True
    assert q1.quote_length <= 32
