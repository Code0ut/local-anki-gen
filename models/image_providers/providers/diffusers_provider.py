"""Diffusers provider -- runs local pipelines via HuggingFace ``diffusers``.

Design highlights
-----------------
* **Lazy loading**: the pipeline is only built inside :meth:`load`, so importing
  this module (or the factory) never pulls in torch/diffusers.
* **Device & dtype auto-selection**: CUDA is detected at load time, with an
  automatic CPU fallback; ``float16`` is used on GPU to save memory.
* **Memory optimisations**: attention slicing, VAE slicing and (when present)
  xFormers are enabled for large resolutions / low-VRAM setups.
* **Model coverage**: SD1.5, SDXL, FLUX, PixArt, Kandinsky via
  ``AutoPipelineForText2Image`` so the same code path handles all families.
* **PIL out**: images are encoded to PNG/JPEG bytes for the unified response.
"""

from __future__ import annotations

from typing import Any

from ..exceptions import ModelLoadError
from ..schemas.request import ImageRequest
from ..schemas.response import ImageResponse, OutputFormat
from ..utils import detect_cuda, has_xformers, pick_dtype
from .base import ImageGenerator


class DiffusersProvider(ImageGenerator):
    provider_name = "diffusers"

    def __init__(
        self,
        model: str | None = "runwayml/stable-diffusion-v1-5",
        device: str | None = None,
        torch_dtype: str | None = None,
        use_xformers: bool | None = None,
        enable_attention_slicing: bool = True,
        enable_vae_slicing: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self._device = device
        self._torch_dtype = torch_dtype
        self._use_xformers = use_xformers
        self._enable_attention_slicing = enable_attention_slicing
        self._enable_vae_slicing = enable_vae_slicing
        self._pipe = None

    # ------------------------------------------------------------------
    def load(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from diffusers import AutoPipelineForText2Image
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ModelLoadError(
                "diffusers provider requires 'diffusers' and 'torch'."
            ) from exc

        cuda = self._device == "cuda" or (
            self._device is None and detect_cuda()
        )
        device = self._device or ("cuda" if cuda else "cpu")
        dtype = self._resolve_dtype(torch, device)

        self._logger.info("Loading %s on %s (%s)", self.model, device, dtype)
        try:
            pipe = AutoPipelineForText2Image.from_pretrained(
                self.model, torch_dtype=dtype
            )
            pipe = pipe.to(device)
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load {self.model}: {exc}"
            ) from exc

        if self._enable_attention_slicing:
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
        if self._enable_vae_slicing and hasattr(pipe, "enable_vae_slicing"):
            try:
                pipe.enable_vae_slicing()
            except Exception:
                pass

        use_xformers = self._use_xformers
        if use_xformers is None:
            use_xformers = cuda and has_xformers()
        if use_xformers:
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                self._logger.warning("xFormers requested but not available.")

        self._pipe = pipe
        self._device = device
        self._loaded = True

    def _resolve_dtype(self, torch: Any, device: str):
        if self._torch_dtype == "float16":
            return torch.float16
        if self._torch_dtype == "float32":
            return torch.float32
        if self._torch_dtype == "bfloat16":
            return torch.bfloat16
        return pick_dtype(device == "cuda")

    def unload(self) -> None:
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        self._loaded = False

    # ------------------------------------------------------------------
    def generate_batch(self, requests: list[ImageRequest]) -> list[ImageResponse]:
        self._ensure_loaded()
        results: list[ImageResponse] = []
        for req in requests:
            pipe_kwargs: dict[str, Any] = dict(
                prompt=req.prompt,
                width=req.width,
                height=req.height,
                num_inference_steps=req.steps,
                guidance_scale=req.guidance_scale,
                num_images_per_prompt=req.num_images,
            )
            if req.negative_prompt:
                pipe_kwargs["negative_prompt"] = req.negative_prompt
            if req.seed is not None:
                pipe_kwargs["generator"] = self._make_generator(req.seed)
            if req.scheduler and hasattr(self._pipe, "scheduler"):
                # Allow overriding the scheduler class where supported.
                pipe_kwargs["scheduler"] = req.scheduler
            pipe_kwargs.update(req.extra_parameters or {})

            out, elapsed = self._time(self._pipe, **pipe_kwargs)

            images = []
            for img in out.images:
                buf = _to_bytes(img, req.output_format)
                images.append(buf)
            results.append(
                ImageResponse(
                    images=images,
                    generation_time=elapsed,
                    provider=self.provider_name,
                    model=self.model or "unknown",
                    seed=req.seed,
                    output_format=req.output_format,
                    metadata={"device": self._device},
                )
            )
        return results

    def _make_generator(self, seed: int):
        import torch

        return torch.Generator(device=self._device).manual_seed(seed)

    def supported_models(self) -> list[str]:
        # Diffusers can load essentially any compatible checkpoint from the
        # Hub or local path; we advertise a few well known families.
        return [
            "runwayml/stable-diffusion-v1-5",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "black-forest-labs/FLUX.1-dev",
            "PixArt-alpha/PixArt-XL-2-1024-MS",
            "kandinsky-community/kandinsky-2-2",
        ]

    def health_check(self) -> bool:
        return self._loaded and self._pipe is not None


def _to_bytes(img, fmt: OutputFormat) -> bytes:
    from io import BytesIO

    buf = BytesIO()
    img.save(buf, format=fmt.upper())
    return buf.getvalue()
