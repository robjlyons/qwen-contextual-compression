from types import SimpleNamespace

import torch

from integration.freetoken_qcc.low_vram.cache_builder import partition_finalized_state


def test_partition_classifies_every_tensor_once():
    state = {
        "model.embed_tokens.weight": torch.ones(2, 3),
        "model.layers.0.mlp.weight": torch.ones(3, 3),
        "model.layers.1.self_attn.weight": torch.ones(3, 3),
        "model.norm.weight": torch.ones(3),
        "lm_head.weight": torch.ones(4, 3),
    }
    embedding, resident, layers = partition_finalized_state(state, 2)
    assert set(embedding) == {"weight"}
    assert set(resident) == {"model.norm.weight", "lm_head.weight"}
    assert set(layers[0]) == {"mlp.weight"}
    assert set(layers[1]) == {"self_attn.weight"}
    assert len(embedding) + len(resident) + sum(map(len, layers)) == len(state)
