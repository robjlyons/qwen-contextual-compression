"""Pure index parsing for selective Hugging Face checkpoint downloads."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Mapping


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


def resolve_checkpoint_tensor_names(
    checkpoint_names: Collection[str], model_names: Collection[str]
) -> dict[str, str]:
    """Map checkpoint keys onto the instantiated Transformers wrapper.

    Some Qwen checkpoints include an extra ``language_model`` component in their
    serialized names even though ``AutoModelForCausalLM.from_config`` returns a
    text-only wrapper whose state dict does not.  Exact names are preferred.  A
    mismatch is resolved only when the longest dotted suffix identifies exactly
    one model tensor, preventing a silent assignment to an ambiguous module.
    """
    available = set(model_names)
    resolved: dict[str, str] = {}
    for checkpoint_name in checkpoint_names:
        if checkpoint_name in available:
            resolved[checkpoint_name] = checkpoint_name
            continue
        parts = checkpoint_name.split(".")
        match = None
        for first_component in range(1, len(parts) - 1):
            suffix = ".".join(parts[first_component:])
            candidates = [name for name in available if name == suffix or name.endswith(f".{suffix}")]
            if len(candidates) == 1:
                match = candidates[0]
                break
            if len(candidates) > 1:
                raise RuntimeError(
                    f"Checkpoint tensor {checkpoint_name!r} ambiguously matches {sorted(candidates)}"
                )
        if match is None:
            raise RuntimeError(
                f"Checkpoint tensor {checkpoint_name!r} has no corresponding tensor in the "
                "official Transformers model instantiated from config"
            )
        resolved[checkpoint_name] = match
    if len(set(resolved.values())) != len(resolved):
        raise RuntimeError("Multiple selected checkpoint tensors resolve to the same model tensor")
    return resolved
