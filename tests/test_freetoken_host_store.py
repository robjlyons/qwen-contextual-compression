import json

import pytest
import torch

from integration.freetoken_qcc.low_vram.host_store import TransformerHostStore
from tests.low_vram_helpers import make_cache


def test_host_store_lazy_layers_metadata_and_neuron_access(tmp_path):
    root = tmp_path / "cache"
    manifest = make_cache(root, [{"weight": torch.arange(12).reshape(3, 4)}, {"weight": torch.ones(2, 4, dtype=torch.float16)}])
    store = TransformerHostStore(root, "fake/model", validate_checksums=True)
    assert store.layer_count == 2 and store._layers == {}
    layer = store.layer(0)
    assert layer.tensor("weight").device.type == "cpu"
    torch.testing.assert_close(layer.neuron_tensor("weight", torch.tensor([2, 0])), layer.tensor("weight")[[2, 0]])
    assert store.host_transformer_bytes == sum(record["tensor_bytes"] for record in manifest["layers"])
    assert store.host_embedding_bytes == manifest["embedding"]["tensor_bytes"]


def test_host_store_rejects_model_and_checksum_mismatch(tmp_path):
    root = tmp_path / "cache"
    make_cache(root, [{"weight": torch.ones(2, 2)}])
    with pytest.raises(ValueError, match="model mismatch"):
        TransformerHostStore(root, "other/model")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["layers"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checksum"):
        TransformerHostStore(root, validate_checksums=True)
