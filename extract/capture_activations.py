"""Stream genuine hidden states entering one FFN to bounded-size shards."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import torch
from extract.extract_ffn import locate_ffn

DEFAULT_PROMPTS = [
 "A quiet river crossed the old city while evening lights appeared.",
 "What causes the seasons on Earth? Explain concisely.",
 "def fibonacci(n):\n    return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
 "If 3x + 7 = 31, solve for x and verify the result.",
 "User: Could you clarify that?\nAssistant: Certainly; let us examine each step.",
 "A train leaves at noon. Reason carefully about when two trains will meet.",
 "A transformer residual stream is normalized before attention and its gated feed-forward network.",
]


class _PartialForwardComplete(Exception):
    """Internal control flow used after the selected official MLP has executed."""


class ActivationShardWriter:
    def __init__(self, output_dir: Path, shard_size: int, storage_dtype: torch.dtype):
        self.output_dir, self.shard_size, self.storage_dtype = output_dir, shard_size, storage_dtype
        output_dir.mkdir(parents=True, exist_ok=True); self.parts = []; self.count = 0; self.index = 0
    def add(self, values: torch.Tensor, token_ids: torch.Tensor) -> None:
        values, token_ids = values.detach().cpu(), token_ids.detach().cpu()
        while len(values):
            room = self.shard_size - sum(len(x[0]) for x in self.parts); take = min(room, len(values))
            self.parts.append((values[:take].to(self.storage_dtype), token_ids[:take])); self.count += take
            values, token_ids = values[take:], token_ids[take:]
            if room == take: self.flush()
    def flush(self) -> None:
        if not self.parts: return
        path = self.output_dir / f"inputs_{self.index:05d}.pt"
        torch.save({"inputs": torch.cat([x[0] for x in self.parts]), "token_ids": torch.cat([x[1] for x in self.parts])}, path)
        self.parts.clear(); self.index += 1


def capture(model, tokenizer, layer_index: int, prompts: list[str], output_dir: Path,
            max_tokens: int, batch_size: int, shard_size: int, seed: int = 42,
            storage_dtype: torch.dtype = torch.float16, exclude_special: bool = True,
            model_name: str = "unknown", revision: str | None = None) -> dict:
    """Capture pre-hook inputs; token filtering is applied using the tokenizer mask."""
    random.Random(seed).shuffle(prompts)
    ffn = locate_ffn(model, layer_index); writer = ActivationShardWriter(output_dir, shard_size, storage_dtype)
    pending_ids: list[torch.Tensor] = []; pending_attention: list[torch.Tensor] = []
    def hook(_module, args):
        hidden = args[0]; ids = pending_ids[0]
        mask = pending_attention[0].bool()
        if exclude_special:
            special = torch.tensor(tokenizer.all_special_ids, device=ids.device)
            if special.numel(): mask &= ~torch.isin(ids, special)
        remaining = max_tokens - writer.count
        writer.add(hidden[mask][:remaining], ids[mask][:remaining])
    handle = ffn.module.register_forward_pre_hook(hook)
    stop_handle = None
    if getattr(model, "_is_selectively_loaded", False):
        def stop_after_official_mlp(_module, _args, _output):
            raise _PartialForwardComplete
        stop_handle = ffn.module.register_forward_hook(stop_after_official_mlp)
    try:
        with torch.inference_mode():
            for start in range(0, len(prompts), batch_size):
                encoded = tokenizer(prompts[start:start+batch_size], return_tensors="pt", padding=True,
                                    truncation=True, max_length=max_tokens)
                device = model.get_input_embeddings().weight.device
                encoded = {k: v.to(device) for k, v in encoded.items()}; pending_ids[:] = [encoded["input_ids"]]
                pending_attention[:] = [encoded.get("attention_mask", torch.ones_like(encoded["input_ids"]))]
                if stop_handle is None:
                    model(**encoded, use_cache=False)
                else:
                    try:
                        model(**encoded, use_cache=False)
                    except _PartialForwardComplete:
                        pass
                if writer.count >= max_tokens: break
    finally:
        handle.remove()
        if stop_handle is not None: stop_handle.remove()
        writer.flush()
    meta = {"model": model_name, "revision": revision, "selected_layer": layer_index,
            "dtype": str(storage_dtype), "hidden_size": ffn.gate_proj.weight.shape[1],
            "captured_vectors": writer.count, "source_prompts": prompts,
            "tokenizer": type(tokenizer).__name__, "exclude_special": exclude_special,
            "timestamp": datetime.now(timezone.utc).isoformat(), "seed": seed, "shards": writer.index,
            "selective_weight_shards": list(getattr(getattr(model, "_selective_load_plan", None), "shard_names", ())) }
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta
