"""Bedrock-backed translation and localization helpers."""

from __future__ import annotations

import logging

from pydantic import BaseModel

from app.agents.factory import create_bedrock_model
from app.domain.models import DetectedLanguage
from app.errors import LocalizationUnavailableError

LOGGER = logging.getLogger(__name__)


class TranslationResult(BaseModel):
    translated_text: str


class LocalizationService:
    """Translate quote text using Bedrock when needed."""

    def __init__(self) -> None:
        self.model = create_bedrock_model()

    def is_available(self) -> bool:
        return self.model is not None

    def require_available(self) -> None:
        if self.model is None:
            raise LocalizationUnavailableError(
                "Localized quote output requires Bedrock, but Bedrock is not available."
            )

    def translate_quote(
        self,
        text: str,
        source_language: str,
        target_language: DetectedLanguage,
    ) -> str:
        """Translate a quote while preserving the original text separately."""

        if target_language.code.lower() == "en":
            return text

        if source_language.lower() == target_language.code.lower():
            return text

        self.require_available()

        try:
            from strands import Agent

            agent = Agent(
                model=self.model,
                system_prompt=(
                    "You translate literary quotations. Return only the translated quote text. "
                    "Do not add commentary, note markers, explanations, or quotation marks unless "
                    "the source requires them."
                ),
            )
            prompt = (
                f"Source language: {source_language}\n"
                f"Target language: {target_language.name} ({target_language.code})\n\n"
                "Translate the following quote faithfully while preserving tone and brevity.\n\n"
                f"{text}"
            )
            result = agent(prompt, structured_output_model=TranslationResult)
            if result.structured_output and result.structured_output.translated_text.strip():
                return result.structured_output.translated_text.strip()
        except Exception as exc:  # pragma: no cover - remote provider specific
            LOGGER.warning("Quote translation failed: %s", exc)
            raise LocalizationUnavailableError(
                f"Bedrock could not translate the selected quotes: {exc}"
            ) from exc

        raise LocalizationUnavailableError("Bedrock did not return a translated quote.")
