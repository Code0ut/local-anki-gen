"""Stability AI provider (uses the REST API)."""

from __future__ import annotations

from typing import Any

from ..exceptions import APIError, ConfigurationError
from ..schemas.request import ImageRequest
from ..schemas.response import ImageResponse
from ..utils import http_get_json
from .base import ImageGenerator

_STABILITY_ENGINES = {
    "sd3": "stable-diffusion-3-large",
    "sdxl": "stable-image-xl-1024-v1-0",
    "core": "stable-image-core-1-0",
    "ultra": "stable-image-ultra-1-0",
}


class StabilityProvider(ImageGenerator):
    provider_name = "stability"

    def __init__(
        self,
        model: str | None = "sd3",
        api_key: str | None = None,
        base_url: str | None = "https://api.stability.ai",
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self._api_key = api_key
        self._base_url = base_url

    def load(self) -> None:
        if self._loaded:
            return
        key = self._api_key or self.config.get("api_key")
        if not key:
            raise ConfigurationError("Stability API key is required.")
        self._key = key
        self._loaded = True

    def _engine(self) -> str:
        return _STABILITY_ENGINES.get(self.model or "sd3", self.model or "sd3")

    def generate_batch(self, requests: list[ImageRequest]) -> list[ImageResponse]:
        import base64

        import requests

        self._ensure_loaded()
        results: list[ImageResponse] = []
        for req in requests:
            url = f"{self._base_url}/v1/generation/{self._engine()}/text-to-image"
            headers = {
                "Authorization": f"Bearer {self._key}",
                "Accept": "application/json",
            }
            body: dict[str, Any] = dict(
                text_prompts=[{"text": req.prompt, "weight": 1.0}],
                width=req.width,
                height=req.height,
                steps=req.steps,
                cfg_scale=req.guidance_scale,
                samples=req.num_images,
                seed=req.seed if req.seed is not None else 0,
                output_format=req.output_format,
            )
            if req.negative_prompt:
                body["text_prompts"].append(
                    {"text": req.negative_prompt, "weight": -1.0}
                )
            body.update(req.extra_parameters or {})

            try:
                resp = requests.post(url, headers=headers, json=body, timeout=120)
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as exc:  # pragma: no cover
                raise APIError(f"Stability API failed: {exc}") from exc

            images = [base64.b64decode(a["base64"]) for a in data["artifacts"]]
            results.append(
                ImageResponse(
                    images=images,
                    generation_time=0.0,
                    provider=self.provider_name,
                    model=self._engine(),
                    seed=req.seed,
                    output_format=req.output_format,
                )
            )
        return results

    def supported_models(self) -> list[str]:
        return list(_STABILITY_ENGINES.keys())

    def health_check(self) -> bool:
        try:
            http_get_json(f"{self._base_url}/v1/user/account", headers={
                "Authorization": f"Bearer {getattr(self, '_key', '')}"
            })
            return True
        except Exception:
            return False
