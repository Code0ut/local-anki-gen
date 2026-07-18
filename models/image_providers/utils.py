"""Shared helpers: logging, image persistence, HTTP, environment detection."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from .schemas.response import ImageResponse, OutputFormat

logger = logging.getLogger("imagegen")


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child logger for a provider."""
    return logging.getLogger(f"imagegen.{name}")


def save_images(
    response: ImageResponse,
    output_dir: str | Path,
    prefix: str = "image",
    fmt: OutputFormat | None = None,
) -> list[Path]:
    """Write every image in ``response`` to ``output_dir`` and return paths."""
    from io import BytesIO

    from PIL import Image

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fmt = fmt or response.output_format or "png"

    paths: list[Path] = []
    for idx, blob in enumerate(response.images, start=1):
        path = out / f"{prefix}_{idx}.{fmt}"
        try:
            img = Image.open(BytesIO(blob))
            img.save(path, format=fmt.upper())
        except Exception:
            # Fall back to raw bytes if PIL cannot decode (e.g. already a file).
            path.write_bytes(blob)
        paths.append(path)
        logger.debug("Saved image to %s", path)
    return paths


def timer() -> Any:
    """Context manager that measures wall-clock seconds."""
    from contextlib import contextmanager

    @contextmanager
    def _t():
        start = time.perf_counter()
        yield
        logger.debug("Elapsed %.3fs", time.perf_counter() - start)

    return _t()


def detect_cuda() -> bool:
    """Return True when a CUDA-capable GPU is visible to PyTorch (if present)."""
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def pick_dtype(cuda: bool) -> Any:
    """Choose a sensible default torch dtype for the device."""
    if not cuda:
        return None  # CPU path uses default float32
    try:
        import torch

        return torch.float16
    except Exception:
        return None


def has_xformers() -> bool:
    """Return True if xFormers is importable."""
    try:
        import xformers  # noqa: F401

        return True
    except Exception:
        return False


def http_get_json(url: str, **kwargs: Any) -> dict[str, Any]:
    """Thin wrapper around ``requests.get`` returning parsed JSON."""
    import requests

    from .exceptions import APIError

    try:
        resp = requests.get(url, timeout=30, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:  # pragma: no cover - network
        raise APIError(f"HTTP GET {url} failed: {exc}") from exc
