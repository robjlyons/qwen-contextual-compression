import pytest

from extract.checkpoint_plan import plan_selective_load


def test_selective_plan_discovers_shards_without_hard_coding_names():
    weight_map = {
        "model.language_model.embed_tokens.weight": "embedding-random-name.safetensors",
        "model.language_model.layers.0.input_layernorm.weight": "early.safetensors",
        "model.language_model.layers.0.self_attn.q_proj.weight": "early.safetensors",
        "model.language_model.layers.0.mlp.gate_proj.weight": "mlp.safetensors",
        "model.language_model.layers.0.mlp.up_proj.weight": "mlp.safetensors",
        "model.language_model.layers.0.mlp.down_proj.weight": "mlp.safetensors",
        "model.language_model.layers.1.mlp.gate_proj.weight": "later.safetensors",
        "lm_head.weight": "head.safetensors",
    }
    plan = plan_selective_load(weight_map, 0)
    assert plan.shard_names == (
        "early.safetensors", "embedding-random-name.safetensors", "mlp.safetensors"
    )
    assert all("layers.1" not in name for name in plan.tensor_names)
    assert "lm_head.weight" not in plan.tensor_names


def test_selective_plan_has_clear_missing_layer_error():
    with pytest.raises(RuntimeError, match="No tensors for language-model layer 3"):
        plan_selective_load({"model.language_model.embed_tokens.weight": "embed.safetensors"}, 3)
