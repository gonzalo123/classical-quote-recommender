"""Helpers to build Strands Agents models."""

from __future__ import annotations

import logging

from settings import (
    BEDROCK_ENDPOINT_URL,
    BEDROCK_MODEL_ID,
    BEDROCK_PROFILE,
    BEDROCK_REGION,
    BEDROCK_TEMPERATURE,
)

LOGGER = logging.getLogger(__name__)


def _normalize_bedrock_model_id(model_id: str) -> str:
    """Normalize model IDs to the regional inference-profile form when needed."""

    if (
        BEDROCK_REGION.startswith("eu-")
        and model_id == "anthropic.claude-sonnet-4-20250514-v1:0"
    ):
        return "eu.anthropic.claude-sonnet-4-20250514-v1:0"
    return model_id


def create_bedrock_model(model_id: str = BEDROCK_MODEL_ID):
    """Return a Bedrock model configured from settings.py."""

    try:
        import boto3
        from strands.models import BedrockModel

        normalized_model_id = _normalize_bedrock_model_id(model_id)

        session = boto3.Session(
            profile_name=BEDROCK_PROFILE or None,
            region_name=BEDROCK_REGION or None,
        )
        return BedrockModel(
            model_id=normalized_model_id,
            boto_session=session,
            endpoint_url=BEDROCK_ENDPOINT_URL or None,
            temperature=BEDROCK_TEMPERATURE,
        )
    except Exception as exc:  # pragma: no cover - dependency/runtime specific
        LOGGER.warning("Could not initialize Bedrock model: %s", exc)
        return None
