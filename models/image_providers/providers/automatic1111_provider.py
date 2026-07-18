"""AUTOMATIC1111 provider (uses the local ``/sdapi/v1/txt2img`` endpoint)."""

from __future__ import annotations

from typing import Any

import requests

from ..exceptions import APIError, ConfigurationError
from ..schemas.request import ImageRequest
from ..schemas.response import ImageResponse
from .base import ImageGenerator


class Automatic1111Provider(ImageGenerator):
    provider_name = "automatic1111"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = "http://127.0.0.1:7860",
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self._base_url = base_url

    def load(self) -> None:
        if self._loaded:
            return
        if not self._base_url:
            raise ConfigurationError("AUTOMATIC1111 base_url is required.")
        try:
            requests.get(f"{self._base_url}/sdapi/v1/sd-models", timeout=5)
        except requests.RequestException as exc:  # pragma: no cover - network
            raise ConfigurationError(
                f"Cannot reach AUTOMATIC1111 at {self._base_url}: {exc}"
            ) from exc
        self._loaded = True

    def generate_batch(self, requests: list[ImageRequest]) -> list[ImageResponse]:
        self._ensure_loaded()
        results: list[ImageResponse] = []
        for req in requests:
            payload: dict[str, Any] = dict(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt or "",
                width=req.width,
                height=req.height,
                steps=req.steps,
                cfg_scale=req.guidance_scale,
                seed=req.seed if req.seed is not None else -1,
                n_iter=1,
                batch_size=req.num_images,
                sampler_name=req.scheduler or "Euler",
                save_images=False,
                send_images=True,
            )
            if self.model:
                payload["override_settings"] = {"sd_model_checkpoint": self.model}
            payload.update(req.extra_parameters or {})

            try:
                r = requests.post(
                    f"{self._base_url}/sdapi/v1/txt2img", json=payload, timeout=300
                )
                r.raise_for_status()
                data = r.json()
            except requests.RequestException as exc:  # pragma: no cover
                raise APIError(f"AUTOMATIC1111 request failed: {exc}") from exc

            import base64

            images = [base64.b64decode(b) for b in data.get("images", [])]
            results.append(
                ImageResponse(
                    images=images,
                    generation_time=data.get("info", {}).get("generation_time", 0.0)
                    if isinstance(data.get("info"), dict) else 0.0,
                    provider=self.provider_name,
                    model=self.model or "a1111-default",
                    seed=req.seed,
                    output_format=req.output_format,
                )
            )
        return results

    def supported_models(self) -> list[str]:
        try:
            r = requests.get(f"{self._base_url}/sdapi/v1/sd-models", timeout=10)
            return [m["model_name"] for m in r.json()]
        except Exception:
            return []

    def health_check(self) -> bool:
        try:
            requests.get(f"{self._base_url}/sdapi/v1/sd-models", timeout=5)
            return True
        except Exception:
            return False
