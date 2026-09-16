import torch

from integration.freetoken_qcc.low_vram.bridge import LowVRAMModelBridge
from integration.freetoken_qcc.low_vram.host_store import TransformerHostStore
from tests.low_vram_helpers import make_cache


class FakeLayer:
    def __init__(self):
        self.weight = torch.empty(0)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight)


def test_bridge_stages_once_per_layer_call_and_restores_host_placeholder(tmp_path):
    root = tmp_path / "cache"
    weights = [torch.randn(4, 4), torch.randn(4, 4)]
    make_cache(root, [{"weight": weight} for weight in weights])
    store = TransformerHostStore(root)
    layers = [FakeLayer(), FakeLayer()]
    placeholders = [layer.weight for layer in layers]
    bridge = LowVRAMModelBridge(store, "cpu")
    bridge.attach_layers(layers)
    x = torch.randn(3, 4)
    output = layers[1].forward(layers[0].forward(x))
    expected = torch.nn.functional.linear(torch.nn.functional.linear(x, weights[0]), weights[1])
    torch.testing.assert_close(output, expected)
    assert all(layer.weight is placeholder for layer, placeholder in zip(layers, placeholders))
    assert bridge.summary()["layer_stages"] == 2
    assert bridge.summary()["layer_prefill_executions"] == 2
    bridge.restore()


def test_bridge_counts_decode_and_preserves_tuple_output(tmp_path):
    root = tmp_path / "cache"
    make_cache(root, [{"weight": torch.eye(4)}])
    store = TransformerHostStore(root)
    layer = FakeLayer()
    original = layer.forward
    layer.forward = lambda x: (original(x), "state")
    bridge = LowVRAMModelBridge(store, "cpu")
    bridge.attach_layers([layer])
    result = layer.forward(torch.ones(1, 4))
    assert result[1] == "state"
    assert bridge.summary()["layer_decode_executions"] == 1
    assert bridge.summary()["finite_outputs"] is True


def test_outer_model_counts_one_forward_not_one_per_layer(tmp_path):
    root = tmp_path / "cache"
    make_cache(root, [{"weight": torch.eye(4)}, {"weight": torch.eye(4)}])
    store = TransformerHostStore(root)
    layers = [FakeLayer(), FakeLayer()]

    class Outer:
        def forward(self, x):
            for layer in layers:
                x = layer.forward(x)
            return x

    outer = Outer()
    bridge = LowVRAMModelBridge(store, "cpu")
    bridge.attach_layers(layers)
    bridge.attach_model_forward(outer)
    outer.forward(torch.ones(1, 4))
    summary = bridge.summary()
    assert summary["layer_decode_executions"] == 2
    assert summary["model_decode_forwards"] == 1
    assert summary["decode_tokens"] == 1
