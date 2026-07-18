"""Provider configuration.

A :class:`ProviderConfig` is the single, validated description of *how* to
build a provider. It can be produced from:

* a plain ``dict`` (:meth:`ProviderConfig.from_dict`),
* a JSON string / file (:meth:`ProviderConfig.from_json`),
* environment variables (:meth:`ProviderConfig.from_env`).

The factory consumes a :class:`ProviderConfig` and is otherwise unaware of
where the settings came from -- this keeps configuration sources decoupled
from provider construction.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ProviderConfig:
    """Resolved configuration used to instantiate a provider."""

    provider: str
    model: str | None = None
    api_key: str | None = None
    # Generic connection settings (used by REST-based providers).
    base_url: str | None = None
    # Provider specific extras (endpoints, model kwargs, ...).
    options: dict[str, Any] = field(default_factory=dict)

    # ---- constructors -------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProviderConfig":
        data = dict(data)
        provider = data.pop("provider", None)
        if not provider:
            raise ValueError("ProviderConfig requires a 'provider' key.")
        return cls(
            provider=provider,
            model=data.pop("model", None),
            api_key=data.pop("api_key", None),
            base_url=data.pop("base_url", None),
            options=data,  # remaining keys are provider-specific
        )

    @classmethod
    def from_json(cls, source: str | Path) -> "ProviderConfig":
        if isinstance(source, Path) or (
            isinstance(source, str) and Path(source).exists()
        ):
            text = Path(source).read_text(encoding="utf-8")
        else:
            text = str(source)
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_env(cls, provider: str, prefix: str = "IMAGEGEN_") -> "ProviderConfig":
        """Build a config from ``IMAGEGEN_PROVIDER=...`` style variables.

        Recognised keys: ``<PREFIX>PROVIDER``, ``<PREFIX>MODEL``,
        ``<PREFIX>API_KEY``, ``<PREFIX>BASE_URL``. Any other ``<PREFIX>XXX``
        variable is passed through as an option (``XXX`` lower-cased).
        """
        opts: dict[str, Any] = {}
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            name = key[len(prefix) :].lower()
            if name == "provider":
                continue  # handled by the explicit arg
            if name in ("model", "api_key", "base_url"):
                opts[name] = value
            else:
                opts[name] = value
        return cls(
            provider=provider,
            model=opts.pop("model", None),
            api_key=opts.pop("api_key", None),
            base_url=opts.pop("base_url", None),
            options=opts,
        )

    # ---- helpers ------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        """Look up an option, then a generic kwarg, then ``default``."""
        return self.options.get(key, default)
