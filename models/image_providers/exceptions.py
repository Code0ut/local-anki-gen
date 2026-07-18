"""ImageGen-specific exception hierarchy.

All errors raised by the library subclass :class:`ImageGenError`, so callers
can catch everything with a single ``except ImageGenError``. Provider-specific
and configuration errors use the more specific subclasses.
"""

from __future__ import annotations


class ImageGenError(Exception):
    """Base class for every error raised by ImageGen."""


class ProviderNotFoundError(ImageGenError):
    """Raised when a requested provider is not registered / not installed."""


class ConfigurationError(ImageGenError):
    """Raised when required configuration (API key, endpoint, ...) is missing."""


class ModelLoadError(ImageGenError):
    """Raised when a model fails to load or is unavailable."""


class GenerationError(ImageGenError):
    """Raised when a generation request fails after the model was loaded."""


class APIError(ImageGenError):
    """Raised when a remote API / HTTP call fails."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
