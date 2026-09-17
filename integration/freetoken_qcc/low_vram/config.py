"""Environment configuration for Phase 5E-A low-VRAM execution."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _flag(value: str, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"0", "1", "false", "true"}:
        raise ValueError(f"{name} must be 0/1/false/true, got {value!r}")
    return normalized in {"1", "true"}


@dataclass(frozen=True)
class LowVRAMConfig:
    enabled: bool = False
    stream_cache: Path | None = None
    asynchronous: bool = False
    slots: int = 2

    @classmethod
    def from_env(cls, environ=None) -> "LowVRAMConfig":
        environment = os.environ if environ is None else environ
        enabled = _flag(environment.get("QCC_FT_LOW_VRAM", "0"), "QCC_FT_LOW_VRAM")
        asynchronous = _flag(environment.get("QCC_FT_STREAM_ASYNC", "0"), "QCC_FT_STREAM_ASYNC")
        cache_value = environment.get("QCC_FT_STREAM_CACHE")
        config = cls(enabled, Path(cache_value).expanduser().resolve() if cache_value else None, asynchronous)
        config.validate(environment.get("QCC_FT_MODE", "off"))
        return config

    def validate(self, sparse_mode: str = "off") -> None:
        if not self.enabled:
            return
        if sparse_mode != "off":
            raise ValueError("Phase 5E-A requires QCC_FT_MODE=off when QCC_FT_LOW_VRAM=1")
        if self.stream_cache is None:
            raise ValueError("QCC_FT_STREAM_CACHE is required when QCC_FT_LOW_VRAM=1")
        if not self.stream_cache.is_dir():
            raise FileNotFoundError(f"QCC_FT_STREAM_CACHE is not a directory: {self.stream_cache}")
        if self.asynchronous:
            raise ValueError("Phase 5E-A supports only QCC_FT_STREAM_ASYNC=0")
        if self.slots != 2:
            raise ValueError("Phase 5E-A requires exactly two staging slots")
