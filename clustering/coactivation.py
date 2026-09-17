"""Compact oracle masks and bounded-memory co-selection graph construction."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from extract.extract_ffn import LocatedFFN, gated_activations
from oracle.importance import importance_scores, topk_indices
from oracle.sweep_sparsity import activation_batches


@dataclass(frozen=True)
class MaskMetadata:
    model: str
    layer: int
    samples: int
    neurons: int
    retention: float
    k: int
    importance: str
    seed: int
    calibration_fraction: float
    storage: str = "neuron-major packed bits"


def count_activation_samples(directory: Path) -> int:
    """Count existing Phase-1 samples without retaining their tensors."""
    total = 0
    for path in sorted(directory.glob("inputs_*.pt")):
        total += len(torch.load(path, map_location="cpu", weights_only=True)["inputs"])
    if total == 0:
        raise FileNotFoundError(f"No Phase-1 inputs_*.pt shards found in {directory}")
    return total


def build_packed_selection_mask(
    ffn: LocatedFFN,
    activation_dir: Path,
    output: Path,
    retention: float,
    importance: str,
    model: str,
    layer: int,
    batch_size: int = 8,
    seed: int = 42,
    calibration_fraction: float = 0.8,
) -> MaskMetadata:
    """Recompute Phase-1 oracle IDs and store N*ceil(T/8) bits, not T*N booleans."""
    if not 0 < retention <= 1 or not 0 < calibration_fraction < 1:
        raise ValueError("retention and calibration_fraction must lie in (0, 1]")
    samples = count_activation_samples(activation_dir)
    neurons = ffn.down_proj.weight.shape[1]
    k = max(1, min(neurons, int(np.ceil(retention * neurons))))
    packed = np.zeros((neurons, (samples + 7) // 8), dtype=np.uint8)
    device, dtype = ffn.gate_proj.weight.device, ffn.gate_proj.weight.dtype
    offset = 0
    with torch.inference_mode():
        for hidden in activation_batches(activation_dir, batch_size):
            activations = gated_activations(hidden.to(device=device, dtype=dtype), ffn)
            scores = importance_scores(activations, ffn.down_proj.weight, importance)
            indices = topk_indices(scores, k).cpu().numpy().astype(np.int32, copy=False)
            token_ids = np.arange(offset, offset + len(indices), dtype=np.int64)
            byte_ids, bits = token_ids // 8, (1 << (token_ids % 8)).astype(np.uint8)
            np.bitwise_or.at(packed, (indices.ravel(), np.repeat(byte_ids, k)), np.repeat(bits, k))
            offset += len(indices)
    metadata = MaskMetadata(model, layer, samples, neurons, retention, k, importance, seed,
                            calibration_fraction)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, packed=packed)
    output.with_suffix(".json").write_text(json.dumps(asdict(metadata), indent=2) + "\n")
    return metadata


def load_mask(path: Path) -> tuple[np.ndarray, MaskMetadata]:
    packed = np.load(path, mmap_mode="r")["packed"]
    metadata = MaskMetadata(**json.loads(path.with_suffix(".json").read_text()))
    if packed.shape != (metadata.neurons, (metadata.samples + 7) // 8):
        raise ValueError(f"Packed mask shape {packed.shape} disagrees with metadata {metadata}")
    return packed, metadata


def unpack_neurons(packed: np.ndarray, start: int, stop: int, samples: int) -> np.ndarray:
    """Unpack a bounded neuron range into boolean selection vectors."""
    return np.unpackbits(packed[start:stop], axis=1, bitorder="little")[:, :samples].astype(bool)


def selection_frequencies(packed: np.ndarray, samples: int) -> np.ndarray:
    lut = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)
    counts = lut[packed].sum(axis=1, dtype=np.int64)
    # Padding bits were initialized to zero.
    return counts / samples


def packed_sample_prefix(packed: np.ndarray, samples: int) -> np.ndarray:
    """Copy only a calibration prefix and clear padding bits in its final byte."""
    result=packed[:,:(samples+7)//8].copy()
    remainder=samples%8
    if remainder: result[:,-1] &= np.uint8((1<<remainder)-1)
    return result


def selection_signatures(
    packed: np.ndarray,
    samples: int,
    dimensions: int = 64,
    seed: int = 42,
    neuron_chunk: int = 256,
    sample_start: int = 0,
    sample_stop: int | None = None,
) -> np.ndarray:
    """Random-projection signatures of calibration-only selection vectors."""
    stop = samples if sample_stop is None else sample_stop
    if not 0 <= sample_start < stop <= samples:
        raise ValueError("invalid sample range")
    rng = np.random.default_rng(seed)
    projection = rng.choice(np.array([-1.0, 1.0], np.float32), size=(stop-sample_start, dimensions))
    signatures = np.empty((len(packed), dimensions), dtype=np.float32)
    for first in range(0, len(packed), neuron_chunk):
        selected = unpack_neurons(packed, first, min(first+neuron_chunk, len(packed)), samples)
        values = selected[:, sample_start:stop].astype(np.float32) @ projection
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        signatures[first:first+len(values)] = values / np.maximum(norms, 1e-12)
    return signatures


def iter_selected_ids(packed: np.ndarray, samples: int, start: int = 0,
                      stop: int | None = None) -> Iterator[np.ndarray]:
    """Yield selected neuron IDs per token without a dense token-neuron matrix."""
    stop = samples if stop is None else stop
    for token in range(start, stop):
        byte, bit = divmod(token, 8)
        yield np.flatnonzero((packed[:, byte] >> bit) & 1).astype(np.int32)
