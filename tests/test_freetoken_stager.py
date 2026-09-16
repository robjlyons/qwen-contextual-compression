from types import SimpleNamespace

import pytest
import torch

from integration.freetoken_qcc.low_vram.host_store import TransformerHostStore
from integration.freetoken_qcc.low_vram.staging import LayerStager
from tests.low_vram_helpers import make_cache


def test_two_slot_stage_activate_restore_and_reuse(tmp_path):
    root = tmp_path / "cache"
    make_cache(root, [{"weight": torch.arange(6).reshape(2, 3)}, {"weight": torch.ones(4, 2, dtype=torch.float16)}])
    store = TransformerHostStore(root)
    stager = LayerStager(store, "cpu")
    layer0 = SimpleNamespace(weight=torch.empty(0))
    layer1 = SimpleNamespace(weight=torch.empty(0))
    original0 = layer0.weight
    stager.stage(0, 0)
    stager.activate(0, 0, layer0)
    assert layer0.weight.dtype == torch.int64 and tuple(layer0.weight.shape) == (2, 3)
    with pytest.raises(RuntimeError, match="active"):
        stager.stage(1, 0)
    stager.deactivate(0)
    assert layer0.weight is original0
    stager.stage(1, 1)
    stager.activate(1, 1, layer1)
    assert layer1.weight.dtype == torch.float16 and tuple(layer1.weight.shape) == (4, 2)
    stager.deactivate(1)
    stager.stage(1, 0)
    assert stager.summary()["layer_stages"] == 3
    assert len(stager.slots) == 2


def test_staged_weight_computation_equals_host_reference(tmp_path):
    root = tmp_path / "cache"
    weight = torch.randn(3, 4)
    make_cache(root, [{"weight": weight}])
    store = TransformerHostStore(root)
    stager = LayerStager(store, "cpu")
    layer = SimpleNamespace(weight=torch.empty(0))
    x = torch.randn(2, 4)
    expected = torch.nn.functional.linear(x, weight)
    stager.stage(0, 0)
    stager.activate(0, 0, layer)
    actual = torch.nn.functional.linear(x, layer.weight)
    stager.deactivate(0)
    torch.testing.assert_close(actual, expected)


def test_runtime_attributes_are_applied_and_restored(tmp_path):
    root = tmp_path / "cache"
    make_cache(root, [{"weight": torch.ones(2, 2)}])
    import json
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["layers"][0]["runtime_attributes"] = {"linear._transposed": True}
    manifest_path.write_text(json.dumps(manifest))
    store = TransformerHostStore(root)
    layer = SimpleNamespace(weight=torch.empty(0), linear=SimpleNamespace(_transposed=False))
    stager = LayerStager(store, "cpu")
    stager.stage(0, 0);stager.activate(0, 0, layer)
    assert layer.linear._transposed is True
    stager.deactivate(0)
    assert layer.linear._transposed is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_staging_preserves_values_dtype_and_bounded_slots(tmp_path):
    root = tmp_path / "cache"
    weight = torch.randn(3, 4, dtype=torch.float16)
    make_cache(root, [{"weight": weight}])
    store = TransformerHostStore(root)
    stager = LayerStager(store, "cuda")
    layer = SimpleNamespace(weight=torch.empty(0))
    stager.stage(0, 0)
    stager.activate(0, 0, layer)
    assert layer.weight.device.type == "cuda" and layer.weight.dtype == torch.float16
    torch.testing.assert_close(layer.weight.cpu(), weight)
    stager.deactivate(0)
    assert len(stager.slots) == 2
