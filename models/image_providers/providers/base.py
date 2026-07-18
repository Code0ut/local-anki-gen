"""Abstract base class every provider implements.

Architectural notes
-------------------
* **Liskov Substitution**: every provider exposes the *same* public surface.
  Code that calls ``generate`` / ``generate_batch`` / ``save`` never needs to
  know which backend produced the pixels.
* **Composition over branching**: the factory wires up a concrete provider
  from configuration -- there is no giant ``if provider == "..."`` block
  anywhere in the call path.
* **Lazy loading**: ``__init__`` only stores configuration. Heavy models are
  loaded on demand in :meth:`load` (or lazily on first :meth:`generate`), so
  constructing a provider is always cheap and import-safe.
* **Extensibility**: adding a provider means dropping a single new module that
  subclasses :class:`ImageGenerator` and registers itself -- no edit to
  existing source required.
"""

from __future__ import annotations

import abc
import time
from typing import Any

from ..exceptions import GenerationError
from ..schemas.request import ImageRequest
from ..schemas.response import ImageResponse
from ..utils import get_logger, timer


class ImageGenerator(abc.ABC):
    """Unified interface for a single image-generation backend."""

    #: Human-friendly provider identifier used by the factory / config.
    provider_name: str = "base"

    def __init__(self, model: str | None = None, **kwargs: Any) -> None:
        self.model = model
        self.config: dict[str, Any] = dict(kwargs)
        self._loaded = False
        self._logger = get_logger(self.provider_name)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def load(self) -> None:
        """Load / initialise any heavy resources (models, clients, ...).

        Implementations set ``self._loaded = True`` on success and raise
        :class:`~imagegen.exceptions.ModelLoadError` on failure.
        """

    def unload(self) -> None:
        """Release resources. Default is a no-op; override when needed."""
        self._loaded = False

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(self, request: ImageRequest) -> ImageResponse:
        """Generate ``request.num_images`` images (convenience wrapper)."""
        return self.generate_batch([request])[0]

    @abc.abstractmethod
    def generate_batch(self, requests: list[ImageRequest]) -> list[ImageResponse]:
        """Generate images for a batch of requests.

        Must return one :class:`~imagegen.schemas.response.ImageResponse` per
        request, preserving order. The default :meth:`generate` delegates here.
        """

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def supported_models(self) -> list[str]:
        """Return the model identifiers this provider can use."""

    def health_check(self) -> bool:
        """Lightweight liveness check. Default: model is loaded.

        Providers talking to a remote service should override this to ping
        the endpoint instead.
        """
        return self._loaded

    # ------------------------------------------------------------------
    # Internal helpers shared by subclasses
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def _time(self, fn, *args, **kwargs) -> tuple[Any, float]:
        """Run ``fn`` and return ``(result, elapsed_seconds)``."""
        with timer():
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
            except GenerationError:
                raise
            except Exception as exc:  # normalise backend noise
                raise GenerationError(
                    f"{self.provider_name} generation failed: {exc}"
                ) from exc
            elapsed = time.perf_counter() - start
        self._logger.info("Generated in %.2fs", elapsed)
        return result, elapsed
