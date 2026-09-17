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
from integration.freetoken_qcc.low_vram.windows_loader import OwnedLoaderDiagnostics


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


def test_loader_only_scalar_input_scale_is_audited_not_spooled(tmp_path, capsys):
    runtime_name = "model.norm.weight"
    metadata_name = "model.layers.0.foo.input_scale"
    building = tmp_path / "cache.building"
    building.mkdir()
    index = spool_normalized_weights(
        iter([
            (metadata_name, torch.tensor(0.125, dtype=torch.float32)),
            (runtime_name, torch.ones(1, dtype=torch.bfloat16)),
        ]),
        _expected([runtime_name]),
        building,
        1,
    )
    ignored = index["ignored_loader_tensors"]
    assert ignored == {
        "count": 1,
        "tensor_bytes": 4,
        "reason_counts": {"runtime-does-not-expose-input-scale": 1},
        "entries": [{
            "name": metadata_name,
            "shape": [],
            "dtype": "torch.float32",
            "tensor_bytes": 4,
            "reason": "runtime-does-not-expose-input-scale",
        }],
    }
    assert len(list((building / "spool").glob("*.safetensors"))) == 1
    assert metadata_name in capsys.readouterr().out


def test_expected_input_scale_is_spooled_as_runtime_state(tmp_path):
    name = "model.layers.0.foo.input_scale"
    expected = {name: ExpectedTensor((), torch.float32, torch.empty((), device="meta"))}
    building = tmp_path / "cache.building"
    building.mkdir()
    index = spool_normalized_weights(iter([(name, torch.tensor(0.125))]), expected, building, 1)
    assert index["ignored_loader_tensors"]["count"] == 0
    assert index["entries"][0]["name"] == name
    assert (building / "spool" / index["entries"][0]["file"]).is_file()


@pytest.mark.parametrize("suffix", ["A_log", "dt_bias"])
def test_gdn_gate_parameter_is_adapted_to_runtime_fp32(tmp_path, suffix, capsys):
    name = f"model.layers.0.linear_attn.{suffix}"
    expected = {
        name: ExpectedTensor((3,), torch.float32, torch.empty(3, device="meta", dtype=torch.float32))
    }
    building = tmp_path / "cache.building"
    building.mkdir()
    index = spool_normalized_weights(
        iter([(name, torch.ones(3, dtype=torch.bfloat16))]), expected, building, 1
    )
    audit = index["normalized_tensor_adaptations"]
    assert audit["count"] == 1
    assert audit["source_bytes"] == 6
    assert audit["target_bytes"] == 12
    assert audit["reason_counts"] == {"gdn-gate-param-runtime-fp32": 1}
    assert audit["entries"][0] == {
        "name": name,
        "shape": [3],
        "source_dtype": "torch.bfloat16",
        "target_dtype": "torch.float32",
        "source_bytes": 6,
        "target_bytes": 12,
        "reason": "gdn-gate-param-runtime-fp32",
    }
    spooled = load_file(building / "spool" / index["entries"][0]["file"])["tensor"]
    assert spooled.dtype == torch.float32
    assert "QCC STREAM TENSOR ADAPTATION" in capsys.readouterr().out


def test_runtime_fp32_a_log_is_not_adapted(tmp_path):
    name = "model.layers.0.linear_attn.A_log"
    expected = {
        name: ExpectedTensor((2,), torch.float32, torch.empty(2, device="meta", dtype=torch.float32))
    }
    building = tmp_path / "cache.building"
    building.mkdir()
    index = spool_normalized_weights(iter([(name, torch.ones(2, dtype=torch.float32))]), expected, building, 1)
    assert index["normalized_tensor_adaptations"] == {
        "count": 0,
        "source_bytes": 0,
        "target_bytes": 0,
        "reason_counts": {},
        "entries": [],
    }


@pytest.mark.parametrize(
    "name, tensor, shape, message",
    [
        ("model.layers.0.linear_attn.some_other_tensor", torch.ones(2, dtype=torch.bfloat16), (2,), "dtype mismatch"),
        ("model.layers.0.linear_attn.A_log", torch.ones(3, dtype=torch.bfloat16), (2,), "shape mismatch"),
        ("model.layers.0.linear_attn.A_log", torch.ones(2, dtype=torch.int32), (2,), "dtype mismatch"),
    ],
)
def test_unproven_runtime_dtype_or_shape_mismatches_fail(tmp_path, name, tensor, shape, message):
    expected = {name: ExpectedTensor(shape, torch.float32, torch.empty(shape, device="meta"))}
    building = tmp_path / "cache.building"
    building.mkdir()
    with pytest.raises(StreamBuildError, match=message):
        spool_normalized_weights(iter([(name, tensor)]), expected, building, 1)


@pytest.mark.parametrize(
    "name, tensor",
    [
        ("model.layers.0.foo.input_scale", torch.ones(2, dtype=torch.float32)),
        ("model.layers.0.foo.input_scale", torch.tensor(1, dtype=torch.float16)),
        ("model.layers.0.foo.mystery_scale", torch.tensor(1, dtype=torch.float32)),
    ],
)
def test_unproven_loader_extras_remain_errors(tmp_path, name, tensor):
    building = tmp_path / "cache.building"
    building.mkdir()
    with pytest.raises(StreamBuildError, match="Unexpected normalized state key"):
        spool_normalized_weights(iter([(name, tensor)]), {}, building, 1)


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
    yield "model.layers.0.projection.input_scale", torch.tensor(0.25, dtype=torch.float32)
    for name in ("model.embed_tokens.weight", "model.layers.0.weight", "model.norm.weight", "lm_head.weight"):
        yield name, torch.ones(1, dtype=torch.bfloat16)


def test_build_finalizes_layer_captures_attribute_and_restores_meta(tmp_path):
    model = FakeModel()
    original = model.layer.weight
    output = tmp_path / "cache"
    diagnostics = OwnedLoaderDiagnostics(
        True,
        "win32",
        loader_mode="lazy-safe-open-clone",
        owned_shards_opened=1,
        owned_shards_released=1,
        owned_source_bytes=123,
    )
    contract = {}
    manifest = build_stream_cache_streaming(
        model, _loaded(), output, "fake", "", "test", {}, contract, loader_diagnostics=diagnostics
    )
    assert model.layer.weight is original
    assert model.layer.weight.is_meta
    assert model.layer._transposed is False
    assert manifest["layers"][0]["runtime_attributes"] == {"_transposed": True}
    assert manifest["ignored_loader_tensors"]["reason_counts"] == {
        "runtime-does-not-expose-input-scale": 1
    }
    assert manifest["normalized_tensor_adaptations"]["count"] == 0
    on_disk_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk_manifest["ignored_loader_tensors"] == manifest["ignored_loader_tensors"]
    assert manifest["source_loader"]["owned_shards_opened"] == 1
    assert manifest["source_loader"]["owned_shards_released"] == 1
    spool_contract = json.loads((output / "freetoken_contract.json").read_text(encoding="utf-8"))
    assert spool_contract["source_loader"] == manifest["source_loader"] == contract["source_loader"]
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
