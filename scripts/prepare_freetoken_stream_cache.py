"""Prepare a finalized CPU FreeToken stream cache without full CUDA materialization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap

from integration.freetoken_qcc.low_vram.preparation import prepare_stream_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default="")
    args = parser.parse_args()
    manifest = prepare_stream_cache(args.model, args.output, args.revision)
    gib = 2**30
    summary = {
        "layers": manifest["layer_count"],
        "embedding_gib": manifest["embedding"]["tensor_bytes"] / gib,
        "resident_gib": manifest["resident"]["tensor_bytes"] / gib,
        "transformer_gib": sum(layer["tensor_bytes"] for layer in manifest["layers"]) / gib,
        "total_tensor_gib": manifest["tensor_bytes"] / gib,
        "total_file_gib": manifest["file_bytes"] / gib,
        "nvfp4_transposed_modules": manifest["nvfp4_transposed_modules"],
        "bf16_tensor_count": manifest["dtype_counts"].get("torch.bfloat16", 0),
        "fp8_scale_tensor_count": manifest["dtype_counts"].get("torch.float8_e4m3fn", 0),
        "int32_packed_tensor_count": manifest["dtype_counts"].get("torch.int32", 0),
        "checksum_status": "written",
    }
    print("STREAM CACHE COMPLETE: " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
