"""Provider factory and registry.

The registry implements the *Open/Closed* principle: adding a provider means
dropping a new module that subclasses :class:`ImageGenerator` and registers
itself via :func:`register_provider` (or by being auto-discovered). No existing
source needs to change.

Two registration paths are supported:

1. **Explicit** -- a provider module calls ``register_provider(cls)`` at import
   time (see the built-in providers below).
2. **Auto-discovery** -- any ``*_provider.py`` module under
   ``imagegen/providers`` that defines a class subclassing
   :class:`ImageGenerator` is imported lazily on first ``create`` call, so new
   providers are picked up automatically.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Any

from .config import ProviderConfig
from .exceptions import ProviderNotFoundError
from .providers.base import ImageGenerator


_REGISTRY: dict[str, type[ImageGenerator]] = {}
_DISCOVERED = False


def register_provider(cls: type[ImageGenerator]) -> None:
    """Register a provider class under its ``provider_name``."""
    name = getattr(cls, "provider_name", None)
    if not name:
        raise ValueError(f"{cls.__name__} must define a 'provider_name'.")
    _REGISTRY[name] = cls


def get_registered() -> dict[str, type[ImageGenerator]]:
    _auto_discover()
    return dict(_REGISTRY)


def _auto_discover() -> None:
    """Lazily import every ``*_provider`` module so they self-register."""
    global _DISCOVERED
    if _DISCOVERED:
        return
    import models.image_providers.providers as pkg

    for mod in pkgutil.iter_modules(pkg.__path__):
        if mod.name.endswith("_provider"):
            try:
                importlib.import_module(f"{pkg.__name__}.{mod.name}")
            except Exception:
                # A provider may fail to import if its optional dependency
                # (torch, openai, ...) is missing -- skip silently.
                continue
    _DISCOVERED = True


# Eagerly register the built-in providers so ``create`` works even before
# auto-discovery runs. Importing these modules only touches stdlib + local
# code (heavy deps are imported lazily inside ``load``).
from .providers.automatic1111_provider import Automatic1111Provider  # noqa: E402
from .providers.comfy_provider import ComfyProvider  # noqa: E402
from .providers.diffusers_provider import DiffusersProvider  # noqa: E402
from .providers.fal_provider import FalProvider  # noqa: E402
from .providers.huggingface_provider import HuggingFaceProvider  # noqa: E402
from .providers.invoke_provider import InvokeProvider  # noqa: E402
from .providers.openai_provider import OpenAIProvider  # noqa: E402
from .providers.replicate_provider import ReplicateProvider  # noqa: E402
from .providers.stability_provider import StabilityProvider  # noqa: E402

for _cls in (
    DiffusersProvider, OpenAIProvider, HuggingFaceProvider, StabilityProvider,
    FalProvider, ReplicateProvider, ComfyProvider, Automatic1111Provider,
    InvokeProvider,
):
    register_provider(_cls)


class ProviderFactory:
    """Builds :class:`ImageGenerator` instances from configuration."""

    @staticmethod
    def create(provider: str | None = None, **kwargs: Any) -> ImageGenerator:
        """Create a provider.

        Accepts either ``provider="name", model="...", api_key="..."`` keyword
        arguments, or a single ``config=ProviderConfig`` / ``config=dict``
        argument. The two styles are mutually exclusive.
        """
        if "config" in kwargs:
            cfg = kwargs["config"]
            if isinstance(cfg, dict):
                cfg = ProviderConfig.from_dict(cfg)
            provider = cfg.provider
            kwargs = dict(
                model=cfg.model,
                api_key=cfg.api_key,
                base_url=cfg.base_url,
                **cfg.options,
            )
        elif provider is None:
            raise ProviderNotFoundError("No provider specified.")

        _auto_discover()
        cls = _REGISTRY.get(provider)
        if cls is None:
            raise ProviderNotFoundError(
                f"Provider '{provider}' is not registered. "
                f"Available: {sorted(_REGISTRY)}"
            )

        # Separate provider constructor kwargs from provider-specific options.
        sig = inspect.signature(cls.__init__)
        accepted = set(sig.parameters) - {"self"}
        init_kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        init_kwargs.setdefault("model", kwargs.get("model"))
        return cls(**init_kwargs)
