import pickle
from types import SimpleNamespace

from integration.freetoken_qcc.bridge import FreeTokenQCCConfig
from integration.freetoken_qcc.worker import (
    FreeTokenIntegrationVersionError,
    install_scheduler_target,
    qcc_scheduler_entry,
)


def original_scheduler(args, ack_queue):
    return args, ack_queue


def test_scheduler_entry_is_top_level_and_pickleable():
    assert qcc_scheduler_entry.__module__ == "integration.freetoken_qcc.worker"
    assert "<locals>" not in qcc_scheduler_entry.__qualname__
    assert pickle.loads(pickle.dumps(qcc_scheduler_entry)) is qcc_scheduler_entry


def test_parent_replaces_only_scheduler_target():
    launch_module = SimpleNamespace(_run_scheduler=original_scheduler, tokenizer="unchanged", detokenizer="unchanged")
    config = FreeTokenQCCConfig(mode="shadow")
    assert install_scheduler_target(config, launch_module) is True
    assert launch_module._run_scheduler is qcc_scheduler_entry
    assert launch_module._qcc_original_scheduler is original_scheduler
    assert launch_module.tokenizer == "unchanged"
    assert launch_module.detokenizer == "unchanged"


def test_off_mode_leaves_scheduler_target_unchanged():
    launch_module = SimpleNamespace(_run_scheduler=original_scheduler)
    assert install_scheduler_target(FreeTokenQCCConfig(), launch_module) is False
    assert launch_module._run_scheduler is original_scheduler
    assert not hasattr(launch_module, "_qcc_original_scheduler")


def test_missing_scheduler_hook_fails_clearly():
    launch_module = SimpleNamespace()
    try:
        install_scheduler_target(FreeTokenQCCConfig(mode="replace"), launch_module)
    except FreeTokenIntegrationVersionError as error:
        assert "_run_scheduler" in str(error)
    else:
        raise AssertionError("missing FreeToken scheduler hook did not fail")


def test_worker_installs_mlp_patch_before_original_scheduler(monkeypatch):
    import integration.freetoken_qcc.bridge as bridge_module
    import integration.freetoken_qcc.worker as worker_module

    events = []
    config = FreeTokenQCCConfig(mode="shadow")

    class FakeBridge:
        def close(self):
            events.append("close")

    def fake_install_patch(received):
        assert received is config
        events.append("install_patch")
        return FakeBridge()

    def fake_scheduler(args, queue):
        events.append(("scheduler", args, queue))
        return "finished"

    monkeypatch.setattr(bridge_module.FreeTokenQCCConfig, "from_env", classmethod(lambda cls: config))
    monkeypatch.setattr(bridge_module, "install_patch", fake_install_patch)
    monkeypatch.setattr(
        worker_module.importlib,
        "import_module",
        lambda name: SimpleNamespace(_run_scheduler=fake_scheduler),
    )
    assert qcc_scheduler_entry("args", "queue") == "finished"
    assert events == ["install_patch", ("scheduler", "args", "queue"), "close"]


def test_worker_refuses_recursive_scheduler_target(monkeypatch):
    import integration.freetoken_qcc.bridge as bridge_module
    import integration.freetoken_qcc.worker as worker_module

    monkeypatch.setattr(
        bridge_module.FreeTokenQCCConfig,
        "from_env",
        classmethod(lambda cls: FreeTokenQCCConfig(mode="shadow")),
    )
    monkeypatch.setattr(
        bridge_module,
        "install_patch",
        lambda config: SimpleNamespace(close=lambda: None),
    )
    monkeypatch.setattr(
        worker_module.importlib,
        "import_module",
        lambda name: SimpleNamespace(_run_scheduler=qcc_scheduler_entry),
    )
    try:
        qcc_scheduler_entry(None, None)
    except FreeTokenIntegrationVersionError as error:
        assert "recursive" in str(error)
    else:
        raise AssertionError("recursive scheduler target did not fail")


def test_launcher_off_mode_does_not_install_scheduler_target(monkeypatch):
    import integration.freetoken_qcc.launch as launcher

    calls = []
    monkeypatch.setattr(launcher.FreeTokenQCCConfig, "from_env", classmethod(lambda cls: FreeTokenQCCConfig()))
    monkeypatch.setattr(launcher, "install_scheduler_target", lambda config: calls.append("patched"))
    monkeypatch.setattr(launcher.importlib, "import_module", lambda name: SimpleNamespace(main=lambda: calls.append("cli")))
    launcher.main()
    assert calls == ["cli"]


def test_launcher_sparse_mode_installs_target_before_cli(monkeypatch):
    import integration.freetoken_qcc.launch as launcher

    calls = []
    config = FreeTokenQCCConfig(mode="shadow")
    monkeypatch.setattr(launcher.FreeTokenQCCConfig, "from_env", classmethod(lambda cls: config))
    monkeypatch.setattr(launcher, "install_scheduler_target", lambda received, force=False: calls.append(("patch", received)))
    monkeypatch.setattr(launcher.importlib, "import_module", lambda name: SimpleNamespace(main=lambda: calls.append(("cli", name))))
    launcher.main()
    assert calls == [("patch", config), ("cli", "freetoken.cli")]


def test_worker_installs_low_vram_adapter_when_sparse_mode_is_off(monkeypatch, tmp_path):
    import integration.freetoken_qcc.bridge as bridge_module
    import integration.freetoken_qcc.low_vram.adapter as adapter_module
    import integration.freetoken_qcc.worker as worker_module

    events = []
    monkeypatch.setenv("QCC_FT_MODE", "off")
    monkeypatch.setenv("QCC_FT_LOW_VRAM", "1")
    monkeypatch.setenv("QCC_FT_STREAM_CACHE", str(tmp_path))
    fake_adapter = SimpleNamespace(close=lambda: events.append("low_close"))
    monkeypatch.setattr(adapter_module, "install_low_vram_adapter", lambda config, expected_model=None: events.append(("low_install", expected_model)) or fake_adapter)
    monkeypatch.setattr(worker_module.importlib, "import_module", lambda name: SimpleNamespace(_run_scheduler=lambda args, queue: events.append("scheduler")))
    qcc_scheduler_entry(SimpleNamespace(model="fake/model"), None)
    assert events == [("low_install", "fake/model"), "scheduler", "low_close"]


def test_worker_uses_canonical_cache_identity_without_mutating_launch_path(
    monkeypatch, tmp_path, capsys
):
    import integration.freetoken_qcc.low_vram.adapter as adapter_module
    import integration.freetoken_qcc.worker as worker_module

    canonical = "RadixArk/Qwen3.8-27B-NVFP4"
    local_path = r"G:\freetoken\models\Qwen3.8-27B-NVFP4"
    monkeypatch.setenv("QCC_FT_MODE", "off")
    monkeypatch.setenv("QCC_FT_LOW_VRAM", "1")
    monkeypatch.setenv("QCC_FT_STREAM_CACHE", str(tmp_path))
    monkeypatch.setenv("QCC_FT_DAEMON_MODEL", canonical)
    monkeypatch.setenv("QCC_FT_DAEMON_MODEL_PATH", local_path)
    observed = {}
    fake_adapter = SimpleNamespace(close=lambda: None)

    def install_adapter(config, expected_model=None):
        observed["cache_source_model"] = expected_model
        return fake_adapter

    monkeypatch.setattr(
        adapter_module,
        "install_low_vram_adapter",
        install_adapter,
    )
    args = SimpleNamespace(model_path=local_path)

    def scheduler(received_args, queue):
        observed["launch_model"] = received_args.model_path

    monkeypatch.setattr(
        worker_module.importlib,
        "import_module",
        lambda name: SimpleNamespace(_run_scheduler=scheduler),
    )

    qcc_scheduler_entry(args, None)

    assert observed == {
        "cache_source_model": canonical,
        "launch_model": local_path,
    }
    assert args.model_path == local_path
    output = capsys.readouterr().out
    assert f"launch_model={local_path}" in output
    assert f"cache_source_model={canonical}" in output
    assert "alias=local-configured-path" in output
