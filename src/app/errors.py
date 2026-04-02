"""Application-specific errors."""

from __future__ import annotations


class LocalizationUnavailableError(RuntimeError):
    """Raised when localized output requires Bedrock and it is unavailable."""
