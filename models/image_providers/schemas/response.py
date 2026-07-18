"""Response schema returned by every provider."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .request import OutputFormat

OutputFormat = Literal["png", "jpeg", "jpg", "webp"]


@dataclass
class ImageResponse:
    """The result of a generation.

    ``images`` holds the raw image bytes together with the format so callers
    can write them to disk or forward them over the network without
    re-encoding.
    """

    images: list[bytes]
    generation_time: float
    provider: str
    model: str
    seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    #: Suggested output format; honoured by :meth:`save` when no explicit
    #: format is supplied.
    output_format: OutputFormat = "png"

    def save(
        self,
        output_dir: str | Path,
        prefix: str = "image",
        fmt: OutputFormat | None = None,
    ) -> list[Path]:
        """Persist every generated image to ``output_dir``.

        Returns the list of written file paths (one per image).
        """
        from ..utils import save_images

        return save_images(self, output_dir=output_dir, prefix=prefix, fmt=fmt)
