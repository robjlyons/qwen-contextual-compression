import gc
import json
import weakref
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from integration.freetoken_qcc.low_vram.streaming_builder import (
    ExpectedTensor,
    StreamBuildError,
    build_stream_cache_streaming,
    classify_key,
    spool_normalized_weights,
)


def _expected(names):
    return {name: ExpectedTensor((1,), torch.bfloat16, torch.empty(1, device="meta", dtype=torch.bfloat16)) for name in names}


def test_spool_streams_without_accumulating_tensors(tmp_path):
    names = ["model.embed_tokens.weight", "model.layers.0.weight", "model.norm.weight", "lm_head.weight"]
    references = []
    maximum_live = 0

    def loaded():
        nonlocal maximum_live
        for index, name in enumerate(names):
            gc.collect()
            maximum_live = max(maximum_live, sum(reference() is not None for reference in references))
            tensor = torch.full((1,), index, dtype=torch.bfloat16)
            references.append(weakref.ref(tensor))
            yield name, tensor

    building = tmp_path / "cache.building"
    building.mkdir()
    index = spool_normalized_weights(loaded(), _expected(names), building, 1)
    gc.collect()
    assert index["tensor_count"] == 4
    assert maximum_live <= 1
    assert sum(reference() is not None for reference in references) == 0
    assert len(list((building / "spool").glob("*.safetensors"))) == 4


@pytest.mark.parametrize(
    "items, message",
    [
        ([('model.norm.weight', torch.ones(1)), ('model.norm.weight', torch.ones(1))], "Duplicate"),
        ([('unknown.weight', torch.ones(1))], "Unexpected"),
        ([], "Missing"),
    ],
)
def test_spool_rejects_invalid_key_sets(tmp_path, items, message):
    building = tmp_path / "cache.building"
    building.mkdir()
    expected = {"model.norm.weight": ExpectedTensor((1,), torch.float32, torch.empty(1, device="meta"))}
    with pytest.raises(StreamBuildError, match=message):
        spool_normalized_weights(iter(items), expected, building, 1)


class FakeFinalizingLayer:
    def __init__(self):
        self.weight = torch.empty(1, device="meta", dtype=torch.bfloat16)
        self._transposed = False

    def state_dict(self):
        return {"weight": self.weight}

    def load_state_dict(self, state):
        self.weight = state.pop("weight")
        self._transposed = True


class FakeModel:
    def __init__(self):
        self.embedding = torch.empty(1, device="meta", dtype=torch.bfloat16)
        self.norm = torch.empty(1, device="meta", dtype=torch.bfloat16)
        self.head = torch.empty(1, device="meta", dtype=torch.bfloat16)
        self.layer = FakeFinalizingLayer()
        self.model = SimpleNamespace(layers=SimpleNamespace(op_list=[self.layer]))

    def state_dict(self):
        return {
            "model.embed_tokens.weight": self.embedding,
            "model.layers.0.weight": self.layer.weight,
            "model.norm.weight": self.norm,
            "lm_head.weight": self.head,
        }


def _loaded():
    for name in ("model.embed_tokens.weight", "model.layers.0.weight", "model.norm.weight", "lm_head.weight"):
        yield name, torch.ones(1, dtype=torch.bfloat16)


def test_build_finalizes_layer_captures_attribute_and_restores_meta(tmp_path):
    model = FakeModel()
    original = model.layer.weight
    output = tmp_path / "cache"
    manifest = build_stream_cache_streaming(model, _loaded(), output, "fake", "", "test", {}, {})
    assert model.layer.weight is original
    assert model.layer.weight.is_meta
    assert model.layer._transposed is False
    assert manifest["layers"][0]["runtime_attributes"] == {"_transposed": True}
    assert load_file(output / "layer_000.safetensors")["weight"].device.type == "cpu"
    assert not (output / "spool").exists()
    assert (output / "manifest.json").is_file()
    assert classify_key("model.layers.0.weight", 1) == ("layer", 0, "weight")


def test_failed_build_leaves_spool_but_no_manifest(tmp_path):
    model = FakeModel()
    model.layer.load_state_dict = lambda state: (_ for _ in ()).throw(RuntimeError("finalizer failed"))
    output = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="finalizer failed"):
        build_stream_cache_streaming(model, _loaded(), output, "fake", "", "test", {}, {})
    building = tmp_path / "cache.building"
    assert building.is_dir()
    assert not (building / "manifest.json").exists()
    assert not output.exists()
