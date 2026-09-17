"""Validated stream-cache manifest schema."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


FORMAT = "qcc-freetoken-stream-cache"
FORMAT_VERSION = 2
REPRESENTATION = "freetoken-finalized-runtime-layout"


def file_sha256(path: Path, chunk_size=8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(root: Path, expected_model: str | None = None) -> dict:
    path = Path(root) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing stream-cache manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT or manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported QCC FreeToken stream-cache format")
    if manifest.get("representation") != REPRESENTATION:
        raise ValueError("Cache is not in finalized FreeToken runtime layout")
    if expected_model is not None and manifest.get("source_model") != expected_model:
        raise ValueError(f"Stream cache model mismatch: {manifest.get('source_model')!r} != {expected_model!r}")
    layers = manifest.get("layers")
    if not isinstance(layers, list) or len(layers) != manifest.get("layer_count"):
        raise ValueError("Manifest layer table does not match layer_count")
    for field in ("source_model", "freetoken_version", "adapter_fingerprint", "tensor_count", "tensor_bytes"):
        if field not in manifest:
            raise ValueError(f"Manifest missing required field: {field}")
    for record in [manifest.get("embedding"), manifest.get("resident"), *layers]:
        if not isinstance(record, dict) or "bindings" not in record:
            raise ValueError("Every cache payload must contain tensor bindings")
    for record in layers:
        if "runtime_attributes" not in record:
            raise ValueError("Every layer payload must contain runtime_attributes")
    return manifest


def validate_cache_files(root: Path, manifest: dict, checksums=True) -> None:
    records = [manifest["embedding"], manifest["resident"], *manifest["layers"]]
    for record in records:
        path = Path(root) / record["file"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing stream-cache payload: {path}")
        if path.stat().st_size != record["file_bytes"]:
            raise ValueError(f"Stream-cache byte-size mismatch: {path}")
        if checksums and file_sha256(path) != record["sha256"]:
            raise ValueError(f"Stream-cache checksum mismatch: {path}")
