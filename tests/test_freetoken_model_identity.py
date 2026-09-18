from integration.freetoken_qcc_identity import resolve_cache_model_identity


MODEL = "RadixArk/Qwen3.8-27B-NVFP4"
LOCAL = r"G:\freetoken\models\Qwen3.8-27B-NVFP4"


def resolve(launch_model, *, path=LOCAL):
    return resolve_cache_model_identity(
        launch_model,
        configured_model=MODEL,
        configured_local_path=path,
        platform="win32",
    )


def test_hub_id_is_canonical_cache_identity():
    assert resolve(MODEL)._asdict() == {
        "launch_model": MODEL,
        "cache_source_model": MODEL,
        "alias": "canonical",
    }


def test_exact_configured_local_path_uses_canonical_cache_identity():
    identity = resolve(LOCAL)
    assert identity.launch_model == LOCAL
    assert identity.cache_source_model == MODEL
    assert identity.alias == "local-configured-path"


def test_windows_case_and_slash_equivalent_path_uses_canonical_identity():
    identity = resolve("g:/freetoken/models/QWEN3.8-27B-NVFP4")
    assert identity.cache_source_model == MODEL
    assert identity.alias == "local-configured-path"


def test_unrelated_local_path_remains_its_own_cache_identity():
    unrelated = r"G:\other\models\Qwen3.8-27B-NVFP4"
    identity = resolve(unrelated)
    assert identity.cache_source_model == unrelated
    assert identity.alias == "launch-model"


def test_basename_only_match_does_not_resolve():
    basename = "Qwen3.8-27B-NVFP4"
    assert resolve(basename).cache_source_model == basename


def test_unset_local_path_preserves_direct_launch_identity():
    launch = r"G:\freetoken\models\Qwen3.8-27B-NVFP4"
    identity = resolve_cache_model_identity(
        launch,
        configured_model=MODEL,
        configured_local_path=None,
        platform="win32",
        environ={},
    )
    assert identity.cache_source_model == launch
    assert identity.alias == "launch-model"
