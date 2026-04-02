"""Candidate selector using Strands Agents with a heuristic fallback."""

from __future__ import annotations

import json
import logging

from app.agents.factory import create_bedrock_model
from app.domain.models import DetectedLanguage, InputAnalysis, RetrievedCandidate, SelectionResult, SelectorChoice
from app.errors import LocalizationUnavailableError
from settings import SELECTOR_CANDIDATE_POOL, FINAL_QUOTE_COUNT, USE_AGENT_SELECTION

LOGGER = logging.getLogger(__name__)


class QuoteSelector:
    """Select the best quotes and generate explanations."""

    def __init__(self) -> None:
        self.model = create_bedrock_model() if USE_AGENT_SELECTION else None

    def _heuristic_reason(self, analysis: InputAnalysis, candidate: RetrievedCandidate) -> str:
        return (
            f"This fragment matches the theme of {analysis.main_theme} and keeps a "
            f"{analysis.tone} tone. It stays compact, reads like a complete quotation, "
            f"and supports the intent to {analysis.intent} without sounding forced."
        )

    def _heuristic_select(
        self,
        analysis: InputAnalysis,
        candidates: list[RetrievedCandidate],
    ) -> SelectionResult:
        ranked = candidates[:FINAL_QUOTE_COUNT]
        choices = [
            SelectorChoice(
                quote_id=candidate.chunk.id,
                why_it_fits=self._heuristic_reason(analysis, candidate),
                score=round(min(candidate.rerank_score, 0.999), 4),
            )
            for candidate in ranked
        ]
        recommended = choices[0].quote_id if choices else ""
        return SelectionResult(recommended_quote_id=recommended, ranked_quotes=choices)

    def select(
        self,
        input_text: str,
        analysis: InputAnalysis,
        candidates: list[RetrievedCandidate],
        detected_language: DetectedLanguage,
    ) -> SelectionResult:
        """Use Strands to choose the best quotes when configured."""

        if self.model is None:
            if detected_language.code != "en":
                raise LocalizationUnavailableError(
                    "Localized explanations require Bedrock when the input language is not English."
                )
            return self._heuristic_select(analysis, candidates)

        try:
            from strands import Agent

            payload = [
                {
                    "quote_id": candidate.chunk.id,
                    "text": candidate.quote_text or candidate.chunk.text,
                    "author": candidate.chunk.author,
                    "work": candidate.chunk.work,
                    "reference": candidate.chunk.reference,
                    "score": candidate.rerank_score,
                }
                for candidate in candidates[:SELECTOR_CANDIDATE_POOL]
            ]
            agent = Agent(
                model=self.model,
                system_prompt=(
                    "You are selecting short quotations from an indexed corpus. "
                    "You must only choose quote_id values that appear in the candidates. "
                    "Prefer one or two complete sentences with a clear beginning and end. "
                    "Rank exactly three options, recommend one, and explain the fit succinctly. "
                    f"Write every explanation in {detected_language.name}."
                ),
            )
            prompt = (
                "Input text:\n"
                f"{input_text}\n\n"
                f"Detected language: {detected_language.name} ({detected_language.code})\n\n"
                "Input analysis:\n"
                f"{analysis.model_dump_json(indent=2)}\n\n"
                "Candidates:\n"
                f"{json.dumps(payload, indent=2)}\n\n"
                "Pick the three best quotes, keeping thematic relevance, tone, clarity, "
                "rhetorical usefulness, and compact sentence-bounded length in mind."
            )
            result = agent(prompt, structured_output_model=SelectionResult)
            selection = result.structured_output
            if selection and selection.ranked_quotes:
                valid_ids = {candidate.chunk.id for candidate in candidates}
                if selection.recommended_quote_id in valid_ids and all(
                    choice.quote_id in valid_ids for choice in selection.ranked_quotes
                ):
                    return selection
        except Exception as exc:  # pragma: no cover - depends on remote provider
            LOGGER.warning("Strands selector failed, using heuristic mode: %s", exc)
            if detected_language.code != "en":
                raise LocalizationUnavailableError(
                    "Localized explanations require a working Bedrock model. "
                    f"Current Bedrock invocation failed: {exc}"
                ) from exc
        return self._heuristic_select(analysis, candidates)
