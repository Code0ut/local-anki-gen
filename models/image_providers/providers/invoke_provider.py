"""InvokeAI provider (uses the local ``/api/v2/`` REST API)."""

from __future__ import annotations

from typing import Any

import requests

from ..exceptions import APIError, ConfigurationError
from ..schemas.request import ImageRequest
from ..schemas.response import ImageResponse
from .base import ImageGenerator


class InvokeProvider(ImageGenerator):
    provider_name = "invokeai"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = "http://127.0.0.1:9090",
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self._base_url = base_url

    def load(self) -> None:
        if self._loaded:
            return
        if not self._base_url:
            raise ConfigurationError("InvokeAI base_url is required.")
        try:
            requests.get(f"{self._base_url}/api/v2/models", timeout=5)
        except requests.RequestException as exc:  # pragma: no cover - network
            raise ConfigurationError(
                f"Cannot reach InvokeAI at {self._base_url}: {exc}"
            ) from exc
        self._loaded = True

    def generate_batch(self, requests: list[ImageRequest]) -> list[ImageResponse]:
        self._ensure_loaded()
        results: list[ImageResponse] = []
        for req in requests:
            payload: dict[str, Any] = dict(
                graph=dict(
                    type="graph",
                    nodes={
                        "main": {
                            "type": "t2i_adapter",  # logical txt2img node
                            "model": self.model or "sd_xl_base_1.0",
                            "positive_prompt": req.prompt,
                            "negative_prompt": req.negative_prompt or "",
                            "width": req.width,
                            "height": req.height,
                            "steps": req.steps,
                            "cfg_scale": req.guidance_scale,
                            "seed": req.seed if req.seed is not None else 0,
                            "num_images": req.num_images,
                        }
                    },
                    edges=[],
                )
            )
            payload.update(req.extra_parameters or {})
            try:
                r = requests.post(
                    f"{self._base_url}/api/v2/generate", json=payload, timeout=300
                )
                r.raise_for_status()
                data = r.json()
            except requests.RequestException as exc:  # pragma: no cover
                raise APIError(f"InvokeAI request failed: {exc}") from exc

            images: list[bytes] = []
            for item in data.get("image_names", []):
                url = f"{self._base_url}/api/v2/images/{item}"
                images.append(requests.get(url, timeout=60).content)
            results.append(
                ImageResponse(
                    images=images,
                    generation_time=0.0,
                    provider=self.provider_name,
                    model=self.model or "invoke-default",
                    seed=req.seed,
                    output_format="png",
                )
            )
        return results

    def supported_models(self) -> list[str]:
        try:
            r = requests.get(f"{self._base_url}/api/v2/models", timeout=10)
            return [m["model_name"] for m in r.json()]
        except Exception:
            return []

    def health_check(self) -> bool:
        try:
            requests.get(f"{self._base_url}/api/v2/models", timeout=5)
            return True
        except Exception:
            return False
