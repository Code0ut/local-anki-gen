"""Replicate provider (uses the official ``replicate`` SDK)."""

from __future__ import annotations

from typing import Any

from ..exceptions import APIError, ConfigurationError
from ..schemas.request import ImageRequest
from ..schemas.response import ImageResponse
from .base import ImageGenerator


class ReplicateProvider(ImageGenerator):
    provider_name = "replicate"

    def __init__(
        self,
        model: str | None = "black-forest-labs/flux-schnell",
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
            import replicate
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ConfigurationError(
                "replicate provider requires 'replicate'."
            ) from exc

        key = self._api_key or self.config.get("api_key")
        if not key:
            raise ConfigurationError("Replicate API token is required.")
        replicate.api_token = key
        self._client = replicate
        self._loaded = True

    def generate_batch(self, requests: list[ImageRequest]) -> list[ImageResponse]:
        import base64

        import requests

        self._ensure_loaded()
        results: list[ImageResponse] = []
        for req in requests:
            # Replicate model versions are referenced by owner/name; extra
            # parameters let callers pass the exact version / params.
            input_dict: dict[str, Any] = dict(
                prompt=req.prompt,
                width=req.width,
                height=req.height,
                num_outputs=req.num_images,
                num_inference_steps=req.steps,
                guidance=req.guidance_scale,
                seed=req.seed if req.seed is not None else 0,
            )
            if req.negative_prompt:
                input_dict["negative_prompt"] = req.negative_prompt
            input_dict.update(req.extra_parameters or {})

            try:
                output = self._client.run(self.model, input=input_dict)
            except Exception as exc:  # pragma: no cover - network
                raise APIError(f"Replicate generation failed: {exc}") from exc

            images: list[bytes] = []
            for uri in output:
                if str(uri).startswith("data:"):
                    images.append(base64.b64decode(str(uri).split(",", 1)[1]))
                else:
                    resp = requests.get(str(uri), timeout=60)
                    resp.raise_for_status()
                    images.append(resp.content)
            results.append(
                ImageResponse(
                    images=images,
                    generation_time=0.0,
                    provider=self.provider_name,
                    model=self.model or "unknown",
                    seed=req.seed,
                    output_format="png",
                    metadata={"output": [str(o) for o in output]},
                )
            )
        return results

    def supported_models(self) -> list[str]:
        return [
            "black-forest-labs/flux-schnell",
            "black-forest-labs/flux-dev",
            "stabilityai/sdxl",
        ]

    def health_check(self) -> bool:
        return self._loaded and self._client is not None
