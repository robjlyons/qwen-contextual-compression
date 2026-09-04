"""Pure index parsing for selective Hugging Face checkpoint downloads."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


INDEX_FILENAME = "model.safetensors.index.json"


@dataclass(frozen=True)
class SelectivePlan:
    """Exact tensors and checkpoint shards selected from an HF weight map."""

    embedding_prefix: str
    layer_prefix: str
    tensor_names: tuple[str, ...]
    shard_names: tuple[str, ...]


def plan_selective_load(weight_map: Mapping[str, str], layer_index: int) -> SelectivePlan:
    """Discover embedding and layer names; never assume checkpoint shard numbers."""
    embedding_candidates = sorted(
        name for name in weight_map
        if name.endswith("language_model.embed_tokens.weight")
        or name.endswith("model.embed_tokens.weight")
    )
    if len(embedding_candidates) != 1:
        raise RuntimeError(
            "Expected exactly one language-model embedding tensor in the safetensors index; "
            f"found {embedding_candidates or '<none>'}."
        )
    backbone_prefix = embedding_candidates[0].removesuffix("embed_tokens.weight")
    layer_prefix = f"{backbone_prefix}layers.{layer_index}."
    layer_names = sorted(name for name in weight_map if name.startswith(layer_prefix))
    if not layer_names:
        raise RuntimeError(f"No tensors for language-model layer {layer_index} were found in {INDEX_FILENAME}.")
    required = tuple(embedding_candidates + layer_names)
    shards = tuple(sorted({weight_map[name] for name in required}))
    return SelectivePlan(
        embedding_prefix=embedding_candidates[0].removesuffix("weight"),
        layer_prefix=layer_prefix, tensor_names=required, shard_names=shards,
    )

