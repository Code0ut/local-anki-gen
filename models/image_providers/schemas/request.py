"""Request/response schemas shared by every provider.

These dataclasses are the *contract* between the caller and any backend.
Providers translate an :class:`ImageRequest` into their own API call and
return an :class:`ImageResponse`, so the rest of an application can stay
completely provider-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

OutputFormat = Literal["png", "jpeg", "jpg", "webp"]


@dataclass
class ImageRequest:
    """A single text-to-image generation request.

    All fields are optional except ``prompt``. Providers ignore fields they
    do not support rather than raising, unless a field is strictly required
    by the backend (e.g. a missing API key), in which case they raise
    :class:`~imagegen.exceptions.ConfigurationError`.
    """

    prompt: str
    negative_prompt: str | None = None
    width: int = 1024
    height: int = 1024
    steps: int = 30
    guidance_scale: float = 7.5
    seed: int | None = None
    num_images: int = 1

    scheduler: str | None = None
    style: str | None = None
    model: str | None = None

    # Advanced / optional controls. Only some providers honour these.
    lora: list[dict[str, Any]] | None = None
    controlnet: list[dict[str, Any]] | None = None

    output_format: OutputFormat = "png"

    # Escape hatch: anything provider specific goes here and is applied
    # directly to the underlying SDK / API call.
    extra_parameters: dict[str, Any] = field(default_factory=dict)
