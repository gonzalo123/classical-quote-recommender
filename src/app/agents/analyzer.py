"""Input analyzer using Strands Agents with a heuristic fallback."""

from __future__ import annotations

import logging
import re

from app.agents.factory import create_bedrock_model
from app.domain.models import AnalyzedInput, DetectedLanguage, InputAnalysis
from app.errors import LocalizationUnavailableError
from settings import USE_AGENT_ANALYSIS

LOGGER = logging.getLogger(__name__)

THEME_RULES = {
    "negotiation": {"middle", "compromise", "agree", "advance", "shared", "point"},
    "feedback": {"feedback", "review", "approach", "focus"},
    "teamwork": {"team", "together", "shared", "advance", "work"},
    "conflict": {"disagree", "not share", "friction", "conflict", "but"},
    "gratitude": {"thanks", "thank", "grateful", "appreciate"},
    "decision": {"decide", "decision", "choose", "next step"},
    "resilience": {"difficult", "hard", "endure", "recover"},
}

SPANISH_HINTS = {
    "gracias",
    "hola",
    "porque",
    "tambien",
    "también",
    "pero",
    "creo",
    "podemos",
    "avanzar",
    "ticket",
    "solicito",
    "validacion",
    "validación",
    "pendiente",
    "un saludo",
}


class AnalyzerResult(AnalyzedInput):
    """Structured analyzer output including detected language."""


class InputAnalyzer:
    """Analyze input text into a structured shape."""

    def __init__(self) -> None:
        self.model = create_bedrock_model() if USE_AGENT_ANALYSIS else None

    def _heuristic_language(self, text: str) -> DetectedLanguage:
        lowered = text.lower()
        if any(token in lowered for token in SPANISH_HINTS) or any(char in lowered for char in "áéíóúñ¿¡"):
            return DetectedLanguage(code="es", name="Spanish")
        return DetectedLanguage(code="en", name="English")

    def _heuristic_analysis(self, text: str) -> InputAnalysis:
        lowered = text.lower()
        themes: list[tuple[str, int]] = []
        for theme, keywords in THEME_RULES.items():
            score = sum(1 for keyword in keywords if keyword in lowered)
            if score > 0:
                themes.append((theme, score))
        themes.sort(key=lambda item: item[1], reverse=True)

        main_theme = themes[0][0] if themes else "communication"
        secondary_themes = [theme for theme, _ in themes[1:4]]

        has_gratitude = any(token in lowered for token in {"thank", "thanks", "appreciate"})
        has_disagreement = any(
            token in lowered
            for token in {"disagree", "not share", "however", "but", "middle ground", "point middle"}
        )
        has_apology = any(token in lowered for token in {"sorry", "regret", "apolog"})

        if has_gratitude and has_disagreement:
            tone = "conciliatory"
            emotion = "guarded optimism"
            intent = "negotiate"
            quote_type = "diplomatic and bridge-building"
        elif has_gratitude:
            tone = "courteous"
            emotion = "gratitude"
            intent = "thank"
            quote_type = "gracious and grounded"
        elif has_disagreement:
            tone = "conciliatory"
            emotion = "controlled tension"
            intent = "negotiate"
            quote_type = "measured and diplomatic"
        elif has_apology:
            tone = "reflective"
            emotion = "regret"
            intent = "apologize"
            quote_type = "humble and restorative"
        elif "?" in text:
            tone = "reflective"
            emotion = "curiosity"
            intent = "reply"
            quote_type = "thoughtful and clarifying"
        else:
            tone = "resolute" if any(token in lowered for token in {"must", "need", "will"}) else "reflective"
            emotion = "focus"
            intent = "persuade" if any(token in lowered for token in {"should", "need", "must"}) else "reply"
            quote_type = "clear and purposeful"

        summary_words = re.sub(r"\s+", " ", text).strip().split()
        summary = " ".join(summary_words[:22]).strip()
        if len(summary_words) > 22:
            summary += "..."

        return InputAnalysis(
            summary=summary or "Short message asking for a rhetorical quote.",
            main_theme=main_theme,
            secondary_themes=secondary_themes,
            tone=tone,
            intent=intent,
            dominant_emotion=emotion,
            recommended_quote_type=quote_type,
        )

    def analyze(self, text: str) -> AnalyzedInput:
        """Run Strands structured output when possible, else use heuristics."""

        if self.model is None:
            detected_language = self._heuristic_language(text)
            if detected_language.code != "en":
                raise LocalizationUnavailableError(
                    "Language-aware analysis requires Bedrock for non-English input."
                )
            return AnalyzedInput(
                detected_language=detected_language,
                input_analysis=self._heuristic_analysis(text),
            )

        try:
            from strands import Agent

            agent = Agent(
                model=self.model,
                system_prompt=(
                    "You analyze short emails or messages for a rhetorical quote recommender. "
                    "First infer the input language from the text itself. "
                    "Return the detected language plus concise, grounded rhetorical analysis. "
                    "Favor themes, tone, intent and rhetorical use. "
                    "Write the analysis fields in the detected language."
                ),
            )
            prompt = (
                "Analyze the following short text for a quote recommendation system.\n\n"
                f"Text:\n{text}\n\n"
                "You must infer the language from the text itself and return both the detected "
                "language and the rhetorical analysis."
            )
            result = agent(prompt, structured_output_model=AnalyzerResult)
            if result.structured_output:
                return result.structured_output
        except Exception as exc:  # pragma: no cover - depends on remote provider
            LOGGER.warning("Strands analyzer failed: %s", exc)
            raise LocalizationUnavailableError(
                "Language-aware analysis requires a working Bedrock model. "
                f"Current Bedrock invocation failed: {exc}"
            ) from exc
