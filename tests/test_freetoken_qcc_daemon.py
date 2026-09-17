import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import integration.freetoken_qcc_daemon as daemon


def _original_builder(model, port, args, *, python, log_dir):
    return [python, "-m", "freetoken.cli", "serve", "--model", model, "--port", str(port), *args], str(
        Path(log_dir) / f"serve-{port}.log"
    )


def test_qcc_model_routes_through_qcc_launch(tmp_path, capsys):
    module = SimpleNamespace(build_serve_command=_original_builder)
    daemon.install_qcc_serve_command_hook("RadixArk/Qwen3.8-27B-NVFP4", module)
    command, log_path = module.build_serve_command(
        "RadixArk/Qwen3.8-27B-NVFP4",
        1919,
        ["--gpu", "0"],
        python="python.exe",
        log_dir=str(tmp_path),
    )
    assert command == [
        "python.exe",
        "-m",
        "integration.freetoken_qcc.launch",
        "serve",
        "--model",
        "RadixArk/Qwen3.8-27B-NVFP4",
        "--port",
        "1919",
        "--gpu",
        "0",
    ]
    assert log_path == str(tmp_path / "serve-1919.log")
    assert "routing model through integration.freetoken_qcc.launch" in capsys.readouterr().out


def test_non_qcc_model_delegates_unchanged(tmp_path):
    calls = []

    def original(model, port, args, *, python, log_dir):
        calls.append(((model, port, args), {"python": python, "log_dir": log_dir}))
        return _original_builder(model, port, args, python=python, log_dir=log_dir)

    module = SimpleNamespace(build_serve_command=original)
    daemon.install_qcc_serve_command_hook("qcc/model", module)
    result = module.build_serve_command(
        "ordinary/model", 1919, ["--gpu", "1"], python="python.exe", log_dir=str(tmp_path)
    )
    assert calls == [
        (("ordinary/model", 1919, ["--gpu", "1"]), {"python": "python.exe", "log_dir": str(tmp_path)})
    ]
    assert result[0][2:4] == ["freetoken.cli", "serve"]


def test_daemon_module_and_hooks_are_torch_free(tmp_path):
    script = r'''
import sys
from types import SimpleNamespace
import integration.freetoken_qcc_daemon as daemon

def builder(model, port, args, *, python, log_dir):
    return [], "log"
def signal_group(pid, sig):
    return "original"
daemon.install_qcc_serve_command_hook("model", SimpleNamespace(build_serve_command=builder))
daemon.install_windows_process_tree_hook(platform="linux", osproc_module=SimpleNamespace(signal_group=signal_group))
assert "torch" not in sys.modules
assert "triton" not in sys.modules
assert "integration.freetoken_qcc" not in sys.modules
'''
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr


def test_existing_cc_is_preserved(monkeypatch):
    environ = {"CC": r"C:\custom\compiler.exe"}
    selected = daemon.configure_windows_triton_compiler(
        platform="win32",
        environ=environ,
        find_spec=lambda name: (_ for _ in ()).throw(AssertionError("find_spec should not run")),
    )
    assert selected == r"C:\custom\compiler.exe"
    assert environ["CC"] == selected


def test_windows_tinycc_is_auto_selected_without_importing_triton(tmp_path, capsys):
    package = tmp_path / "triton"
    compiler = package / "runtime" / "tcc" / "tcc.exe"
    compiler.parent.mkdir(parents=True)
    compiler.write_bytes(b"tinycc")
    environ = {}
    spec = SimpleNamespace(origin=str(package / "__init__.py"), submodule_search_locations=[str(package)])
    selected = daemon.configure_windows_triton_compiler(
        platform="win32", environ=environ, find_spec=lambda name: spec
    )
    assert selected == str(compiler)
    assert environ == {"CC": str(compiler)}
    assert str(compiler) in capsys.readouterr().out


@pytest.mark.parametrize(
    "sig, expected",
    [
        (signal.SIGTERM, ["taskkill", "/PID", "4321", "/T"]),
        (signal.SIGKILL, ["taskkill", "/PID", "4321", "/T", "/F"]),
    ],
)
def test_windows_tree_shutdown_targets_only_owned_pid(sig, expected):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    daemon.windows_signal_tree(4321, sig, run=run, process_exists=lambda pid: True)
    assert calls[0][0] == expected
    assert "/IM" not in calls[0][0]
    assert calls[0][0].count("4321") == 1


def test_non_windows_lifecycle_symbol_is_untouched():
    calls = []

    def original(pid, sig):
        calls.append((pid, sig))
        return "delegated"

    module = SimpleNamespace(signal_group=original)
    returned = daemon.install_windows_process_tree_hook(platform="linux", osproc_module=module)
    assert returned is original
    assert module.signal_group is original
    assert module.signal_group(10, signal.SIGTERM) == "delegated"
    assert calls == [(10, signal.SIGTERM)]


def test_already_vanished_windows_process_is_success():
    result = SimpleNamespace(returncode=128, stdout="not found", stderr="")
    daemon.windows_signal_tree(
        99,
        signal.SIGTERM,
        run=lambda *args, **kwargs: result,
        process_exists=lambda pid: False,
    )


def test_missing_daemon_symbols_fail_clearly():
    with pytest.raises(daemon.QCCDaemonCompatibilityError, match="QCC FreeToken daemon compatibility error"):
        daemon.install_qcc_serve_command_hook("model", SimpleNamespace())
    with pytest.raises(daemon.QCCDaemonCompatibilityError, match="QCC FreeToken daemon compatibility error"):
        daemon.install_windows_process_tree_hook(platform="win32", osproc_module=SimpleNamespace())
    with pytest.raises(daemon.QCCDaemonCompatibilityError, match="QCC FreeToken daemon compatibility error"):
        daemon._daemon_main(SimpleNamespace(), None)
