import pytest


def test_launcher_installs_worker_for_low_vram_off_mode(monkeypatch, tmp_path):
    import integration.freetoken_qcc.launch as launcher

    monkeypatch.setenv("QCC_FT_MODE", "off")
    monkeypatch.setenv("QCC_FT_LOW_VRAM", "1")
    monkeypatch.setenv("QCC_FT_STREAM_CACHE", str(tmp_path))
    calls = []
    monkeypatch.setattr(launcher, "install_scheduler_target", lambda config, force=False: calls.append((config.mode, force)))
    monkeypatch.setattr(launcher.importlib, "import_module", lambda name: type("CLI", (), {"main": staticmethod(lambda: calls.append("cli"))}))
    launcher.main()
    assert calls == [("off", True), "cli"]
