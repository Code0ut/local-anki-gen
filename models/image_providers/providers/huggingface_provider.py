"""HuggingFace Inference API provider (uses ``huggingface_hub.InferenceClient``)."""

from __future__ import annotations

from typing import Any

from ..exceptions import APIError, ConfigurationError
from ..schemas.request import ImageRequest
from ..schemas.response import ImageResponse
from .base import ImageGenerator


class HuggingFaceProvider(ImageGenerator):
    provider_name = "huggingface"

    def __init__(
        self,
        model: str | None = "stabilityai/stable-diffusion-xl-base-1.0",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self._api_key = api_key
        self._client = None

    def load(self) -> None:
        if self._loaded:
            return
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ConfigurationError(
                "huggingface provider requires 'huggingface_hub'."
            ) from exc

        key = self._api_key or self.config.get("api_key")
        if not key:
            raise ConfigurationError("HuggingFace API token is required.")
        self._client = InferenceClient(token=key)
        self._loaded = True

    def generate_batch(self, requests: list[ImageRequest]) -> list[ImageResponse]:
        self._ensure_loaded()
        results: list[ImageResponse] = []
        for req in requests:
            params: dict[str, Any] = dict(
                prompt=req.prompt,
                height=req.height,
                width=req.width,
                num_inference_steps=req.steps,
                guidance_scale=req.guidance_scale,
                seed=req.seed if req.seed is not None else 0,
            )
            if req.negative_prompt:
                params["negative_prompt"] = req.negative_prompt
            params.update(req.extra_parameters or {})

            try:
                images = []
                for _ in range(req.num_images):
                    img_bytes = self._client.text_to_image(
                        self.model, **params
                    ).tobytes()
                    images.append(img_bytes)
            except Exception as exc:  # pragma: no cover - network
                raise APIError(f"HuggingFace generation failed: {exc}") from exc

            results.append(
                ImageResponse(
                    images=images,
                    generation_time=0.0,
                    provider=self.provider_name,
                    model=self.model or "unknown",
                    seed=req.seed,
                    output_format="png",
                )
            )
        return results

    def supported_models(self) -> list[str]:
        return [
            "stabilityai/stable-diffusion-xl-base-1.0",
            "runwayml/stable-diffusion-v1-5",
            "black-forest-labs/FLUX.1-dev",
        ]

    def health_check(self) -> bool:
        return self._loaded and self._client is not None
