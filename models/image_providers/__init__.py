"""ImageGen - a unified, provider-agnostic interface for text-to-image generation.

ImageGen is designed in the spirit of LiteLLM, but for image generation.
Switch providers (Diffusers, OpenAI, Stability, Fal, Replicate, HuggingFace,
ComfyUI, AUTOMATIC1111, InvokeAI, ...) by changing only the provider name or
configuration -- the rest of the application never learns *which* backend
produced the image.
"""

from .factory import ProviderFactory
from .config import ProviderConfig
from .exceptions import (
    ImageGenError,
    ProviderNotFoundError,
    ModelLoadError,
    GenerationError,
    APIError,
    ConfigurationError,
)
from .schemas.request import ImageRequest
from .schemas.response import ImageResponse

__version__ = "0.1.0"

__all__ = [
    "ProviderFactory",
    "ImageRequest",
    "ImageResponse",
    "ImageGenError",
    "ProviderNotFoundError",
    "ModelLoadError",
    "GenerationError",
    "APIError",
    "ConfigurationError",
    "ProviderConfig",
]
