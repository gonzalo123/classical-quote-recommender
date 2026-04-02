from app.domain.models import AnalyzedInput, CorpusChunk, DetectedLanguage, InputAnalysis, RetrievedCandidate, SelectionResult, SelectorChoice
from app.errors import LocalizationUnavailableError
from app.services.localization_service import LocalizationService
from app.services.quote_service import QuoteService


def test_quote_response_preserves_original_and_adds_translation(monkeypatch) -> None:
    service = QuoteService()

    chunk = CorpusChunk(
        id="q1",
        text="Sing, goddess, the anger of Achilles.",
        work="The Iliad",
        author="Homer",
        reference="section 1",
        length=6,
        metadata={"language": "en"},
    )
    candidate = RetrievedCandidate(chunk=chunk, rerank_score=0.91)

    monkeypatch.setattr(
        service.localization_service,
        "is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        service.analyzer,
        "analyze",
        lambda text: AnalyzedInput(
            detected_language=DetectedLanguage(code="es", name="Spanish"),
            input_analysis=InputAnalysis(
                summary="Resumen breve.",
                main_theme="negociacion",
                secondary_themes=["equipo"],
                tone="conciliador",
                intent="negociar",
                dominant_emotion="calma",
                recommended_quote_type="diplomatica",
            ),
        ),
    )
    monkeypatch.setattr(service, "_load_index_if_needed", lambda: None)
    service._vector_store = object()
    monkeypatch.setattr(
        "app.retrieval.hybrid.HybridRetriever.retrieve",
        lambda self, text, analysis, top_k: [candidate],
    )
    monkeypatch.setattr(
        service.reranker,
        "rerank",
        lambda text, analysis, candidates: candidates,
    )
    monkeypatch.setattr(
        service.selector,
        "select",
        lambda input_text, analysis, candidates, detected_language: SelectionResult(
            recommended_quote_id="q1",
            ranked_quotes=[
                SelectorChoice(
                    quote_id="q1",
                    why_it_fits="Encaja por tono y tema.",
                    score=0.91,
                )
            ],
        ),
    )
    monkeypatch.setattr(
        service.localization_service,
        "translate_quote",
        lambda text, source_language, target_language: "Canta, diosa, la ira de Aquiles.",
    )

    result = service.quote("Gracias por tu feedback. Creo que podemos avanzar.")

    assert result.detected_language.code == "es"
    assert result.recommended_quote.text == "Sing, goddess, the anger of Achilles."
    assert result.recommended_quote.translated_text == "Canta, diosa, la ira de Aquiles."


def test_quote_fails_clearly_when_localized_output_requires_bedrock(monkeypatch) -> None:
    service = QuoteService()
    monkeypatch.setattr(
        service.analyzer,
        "analyze",
        lambda text: (_ for _ in ()).throw(
            LocalizationUnavailableError("Bedrock is required for non-English input.")
        ),
    )
    monkeypatch.setattr(service, "_load_index_if_needed", lambda: None)
    service._vector_store = object()

    try:
        service.quote("Gracias por tu ayuda.")
    except LocalizationUnavailableError as exc:
        assert "Bedrock" in str(exc)
    else:
        raise AssertionError("Expected LocalizationUnavailableError")


def test_localization_service_skips_translation_when_target_language_is_english() -> None:
    service = LocalizationService()

    translated = service.translate_quote(
        text="Sing, goddess, the anger of Achilles.",
        source_language="en",
        target_language=DetectedLanguage(code="en", name="English"),
    )

    assert translated == "Sing, goddess, the anger of Achilles."
