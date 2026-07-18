"""OpenAI Images API provider (uses the official ``openai`` SDK)."""

from __future__ import annotations

from typing import Any

from ..exceptions import APIError, ConfigurationError
from ..schemas.request import ImageRequest
from ..schemas.response import ImageResponse
from .base import ImageGenerator


class OpenAIProvider(ImageGenerator):
    provider_name = "openai"

    def __init__(
        self,
        model: str | None = "dall-e-3",
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
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ConfigurationError("openai provider requires 'openai'.") from exc

        key = self._api_key or self.config.get("api_key")
        if not key:
            raise ConfigurationError("OpenAI API key is required.")
        self._client = OpenAI(api_key=key)
        self._loaded = True

    def _params(self, req: ImageRequest) -> dict[str, Any]:
        size = f"{req.width}x{req.height}"
        params: dict[str, Any] = dict(
            model=self.model or "dall-e-3",
            prompt=req.prompt,
            n=req.num_images,
            size=size,
            response_format="b64_json",
        )
        if req.style in ("vivid", "natural"):
            params["style"] = req.style
        if self.model == "gpt-image-1":
            params.pop("style", None)
        params.update(req.extra_parameters or {})
        return params

    def generate_batch(self, requests: list[ImageRequest]) -> list[ImageResponse]:
        self._ensure_loaded()
        import base64

        results: list[ImageResponse] = []
        for req in requests:
            params = self._params(req)
            try:
                resp = self._client.images.generate(**params)
            except Exception as exc:  # pragma: no cover - network
                raise APIError(f"OpenAI generation failed: {exc}") from exc

            images = [base64.b64decode(d.b64_json) for d in resp.data]
            results.append(
                ImageResponse(
                    images=images,
                    generation_time=0.0,
                    provider=self.provider_name,
                    model=self.model or "unknown",
                    seed=req.seed,
                    output_format="png",
                    metadata={"created": getattr(resp, "created", None)},
                )
            )
        return results

    def supported_models(self) -> list[str]:
        return ["dall-e-2", "dall-e-3", "gpt-image-1"]

    def health_check(self) -> bool:
        return self._loaded and self._client is not None
