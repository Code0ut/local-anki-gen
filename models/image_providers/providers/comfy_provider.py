"""ComfyUI provider (communicates over the local REST API).

Flow: POST a workflow/prompt to ``/prompt``, poll ``/history`` for the job id,
then download each produced image from ``/view``.
"""

from __future__ import annotations

from typing import Any

import requests

from ..exceptions import APIError, ConfigurationError
from ..schemas.request import ImageRequest
from ..schemas.response import ImageResponse
from .base import ImageGenerator

# A minimal SDXL txt2img graph. ``{seed}`` / ``{prompt}`` etc. are filled in
# per request. This keeps the example self-contained without shipping JSON.
_TEMPLATE = {
    "3": {"class_type": "KSampler", "inputs": {
        "seed": 0, "steps": 20, "cfg": 7.0, "sampler_name": "euler",
        "scheduler": "normal", "denoise": 1.0,
        "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0],
        "latent_image": ["5", 0]}},
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {
        "ckpt_name": "sd_xl_base_1.0.safetensors"}},
    "5": {"class_type": "EmptyLatentImage", "inputs": {
        "width": 1024, "height": 1024, "batch_size": 1}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["4", 1]}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["4", 1]}},
    "8": {"class_type": "VAEDecode", "inputs": {
        "samples": ["3", 0], "vae": ["4", 2]}},
    "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0]}},
}


class ComfyProvider(ImageGenerator):
    provider_name = "comfyui"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = "http://127.0.0.1:8188",
        client_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self._base_url = base_url
        self._client_id = client_id or "imagegen"

    def load(self) -> None:
        if self._loaded:
            return
        if not self._base_url:
            raise ConfigurationError("ComfyUI base_url is required.")
        # Validate reachability.
        try:
            requests.get(f"{self._base_url}/system_stats", timeout=5)
        except requests.RequestException as exc:  # pragma: no cover - network
            raise ConfigurationError(
                f"Cannot reach ComfyUI at {self._base_url}: {exc}"
            ) from exc
        self._loaded = True

    def _build_prompt(self, req: ImageRequest) -> dict[str, Any]:
        import copy

        graph = copy.deepcopy(_TEMPLATE)
        graph["3"]["inputs"].update(
            seed=req.seed if req.seed is not None else 0,
            steps=req.steps,
            cfg=req.guidance_scale,
        )
        if req.scheduler:
            graph["3"]["inputs"]["scheduler"] = req.scheduler
        graph["5"]["inputs"].update(width=req.width, height=req.height,
                                    batch_size=req.num_images)
        graph["6"]["inputs"]["text"] = req.prompt
        graph["7"]["inputs"]["text"] = req.negative_prompt or ""
        if self.model:
            graph["4"]["inputs"]["ckpt_name"] = self.model
        return {"prompt": graph, "client_id": self._client_id}

    def generate_batch(self, requests: list[ImageRequest]) -> list[ImageResponse]:
        self._ensure_loaded()
        results: list[ImageResponse] = []
        for req in requests:
            try:
                r = requests.post(
                    f"{self._base_url}/prompt", json=self._build_prompt(req),
                    timeout=30,
                )
                r.raise_for_status()
                prompt_id = r.json()["prompt_id"]
                outputs = self._wait_and_collect(prompt_id, req)
            except requests.RequestException as exc:  # pragma: no cover
                raise APIError(f"ComfyUI request failed: {exc}") from exc

            results.append(
                ImageResponse(
                    images=outputs,
                    generation_time=0.0,
                    provider=self.provider_name,
                    model=self.model or "comfy-default",
                    seed=req.seed,
                    output_format="png",
                )
            )
        return results

    def _wait_and_collect(self, prompt_id: str, req: ImageRequest) -> list[bytes]:
        import time

        deadline = time.time() + 600
        while time.time() < deadline:
            r = requests.get(f"{self._base_url}/history/{prompt_id}", timeout=30)
            r.raise_for_status()
            data = r.json()
            if prompt_id in data:
                images: list[bytes] = []
                for node in data[prompt_id]["outputs"].values():
                    for img in node.get("images", []):
                        url = (
                            f"{self._base_url}/view?filename={img['filename']}"
                            f"&subfolder={img.get('subfolder', '')}"
                            f"&type={img.get('type', '')}"
                        )
                        images.append(requests.get(url, timeout=60).content)
                return images
            time.sleep(1)
        raise APIError("ComfyUI job timed out.")  # pragma: no cover

    def supported_models(self) -> list[str]:
        try:
            r = requests.get(f"{self._base_url}/object_info", timeout=10)
            info = r.json()
            return list(info.get("CheckpointLoaderSimple", {})
                         .get("input", {}).get("required", {})
                         .get("ckpt_name", [[]])[0])
        except Exception:
            return []

    def health_check(self) -> bool:
        try:
            requests.get(f"{self._base_url}/system_stats", timeout=5)
            return True
        except Exception:
            return False
