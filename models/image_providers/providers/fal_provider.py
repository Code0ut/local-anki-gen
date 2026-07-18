"""Fal.ai provider (uses the official ``fal-client`` SDK)."""

from __future__ import annotations

from typing import Any

from ..exceptions import APIError, ConfigurationError
from ..schemas.request import ImageRequest
from ..schemas.response import ImageResponse
from .base import ImageGenerator


class FalProvider(ImageGenerator):
    provider_name = "fal"

    def __init__(
        self,
        model: str | None = "fal-ai/flux/dev",
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
            import fal_client
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ConfigurationError("fal provider requires 'fal-client'.") from exc

        key = self._api_key or self.config.get("api_key")
        if not key:
            raise ConfigurationError("Fal API key is required.")
        self._client = fal_client
        self._client.api_key = key
        self._loaded = True

    def generate_batch(self, requests: list[ImageRequest]) -> list[ImageResponse]:
        import base64

        import requests

        self._ensure_loaded()
        results: list[ImageResponse] = []
        for req in requests:
            arguments: dict[str, Any] = dict(
                prompt=req.prompt,
                num_images=req.num_images,
                seed=req.seed if req.seed is not None else 0,
                enable_safety_checker=False,
            )
            if req.negative_prompt:
                arguments["negative_prompt"] = req.negative_prompt
            arguments.update(req.extra_parameters or {})

            try:
                result = self._client.run(self.model, arguments=arguments)
            except Exception as exc:  # pragma: no cover - network
                raise APIError(f"Fal generation failed: {exc}") from exc

            images: list[bytes] = []
            for item in result.get("images", []):
                url = item["url"]
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                images.append(resp.content)
            results.append(
                ImageResponse(
                    images=images,
                    generation_time=result.get("timings", {}).get("inference", 0.0),
                    provider=self.provider_name,
                    model=self.model or "unknown",
                    seed=req.seed,
                    output_format="png",
                    metadata=result,
                )
            )
        return results

    def supported_models(self) -> list[str]:
        return ["fal-ai/flux/dev", "fal-ai/flux/schnell", "fal-ai/sd35"]

    def health_check(self) -> bool:
        return self._loaded and self._client is not None
