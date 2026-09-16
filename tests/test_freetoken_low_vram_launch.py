import pytest


def test_launcher_refuses_unvalidated_loader_adapter(monkeypatch, tmp_path):
    import integration.freetoken_qcc.launch as launcher

    monkeypatch.setenv("QCC_FT_MODE", "off")
    monkeypatch.setenv("QCC_FT_LOW_VRAM", "1")
    monkeypatch.setenv("QCC_FT_STREAM_CACHE", str(tmp_path))
    monkeypatch.setattr(launcher.importlib, "import_module", lambda name: (_ for _ in ()).throw(AssertionError("FreeToken must not start")))
    with pytest.raises(RuntimeError, match="no adapter"):
        launcher.main()
