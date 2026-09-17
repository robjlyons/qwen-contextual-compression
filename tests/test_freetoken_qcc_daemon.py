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


LOCAL_MODEL = (
    r"C:\Users\Rob\.cache\huggingface\hub\models--RadixArk--Qwen3.8-27B-NVFP4"
    r"\snapshots\319f741cce68d7914884900c138a1fbb70a42f30"
)


def _route_command(module, model, tmp_path):
    return module.build_serve_command(
        model, 1919, ["--gpu", "0"], python="python.exe", log_dir=str(tmp_path)
    )[0]


def test_exact_configured_local_path_routes_and_is_preserved(tmp_path, capsys):
    module = SimpleNamespace(build_serve_command=_original_builder)
    daemon.install_qcc_serve_command_hook(
        "RadixArk/Qwen3.8-27B-NVFP4",
        module,
        platform="win32",
        environ={"QCC_FT_DAEMON_MODEL_PATH": LOCAL_MODEL},
    )
    command = _route_command(module, LOCAL_MODEL, tmp_path)
    assert command[2:4] == ["integration.freetoken_qcc.launch", "serve"]
    assert command[command.index("--model") + 1] == LOCAL_MODEL
    assert "routing configured local model" in capsys.readouterr().out


def test_windows_case_and_separator_equivalent_local_path_routes(tmp_path):
    module = SimpleNamespace(build_serve_command=_original_builder)
    daemon.install_qcc_serve_command_hook(
        "RadixArk/Qwen3.8-27B-NVFP4",
        module,
        configured_local_path=LOCAL_MODEL,
        platform="win32",
    )
    equivalent = LOCAL_MODEL.upper().replace("\\", "/")
    command = _route_command(module, equivalent, tmp_path)
    assert command[2:4] == ["integration.freetoken_qcc.launch", "serve"]
    assert command[command.index("--model") + 1] == equivalent


@pytest.mark.parametrize(
    "candidate",
    [
        r"C:\models\unrelated-model",
        r"C:\other\319f741cce68d7914884900c138a1fbb70a42f30",
        "319f741cce68d7914884900c138a1fbb70a42f30",
    ],
)
def test_unrelated_or_basename_only_local_path_delegates(tmp_path, candidate):
    module = SimpleNamespace(build_serve_command=_original_builder)
    daemon.install_qcc_serve_command_hook(
        "RadixArk/Qwen3.8-27B-NVFP4",
        module,
        configured_local_path=LOCAL_MODEL,
        platform="win32",
    )
    command = _route_command(module, candidate, tmp_path)
    assert command[2:4] == ["freetoken.cli", "serve"]


def test_unset_local_path_preserves_hub_id_only_behavior(tmp_path):
    module = SimpleNamespace(build_serve_command=_original_builder)
    daemon.install_qcc_serve_command_hook(
        "RadixArk/Qwen3.8-27B-NVFP4", module, platform="win32", environ={}
    )
    assert _route_command(module, LOCAL_MODEL, tmp_path)[2:4] == ["freetoken.cli", "serve"]
    assert _route_command(module, "RadixArk/Qwen3.8-27B-NVFP4", tmp_path)[2:4] == [
        "integration.freetoken_qcc.launch",
        "serve",
    ]


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
        (9, ["taskkill", "/PID", "4321", "/T", "/F"]),
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


def test_graceful_tree_signal_failure_while_alive_allows_escalation(capsys):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=128, stdout="", stderr="unsupported")

    daemon.windows_signal_tree(
        4321, signal.SIGTERM, run=run, process_exists=lambda pid: True
    )
    assert calls == [["taskkill", "/PID", "4321", "/T"]]
    assert "/F" not in calls[0]
    output = capsys.readouterr().out
    assert "QCC WINDOWS GRACEFUL TREE SIGNAL UNSUPPORTED" in output
    assert "pid=4321 rc=128" in output
    assert "allowing FreeToken grace-period escalation" in output


def test_forced_tree_signal_failure_while_alive_still_raises():
    result = SimpleNamespace(returncode=128, stdout="", stderr="access denied")
    with pytest.raises(subprocess.CalledProcessError) as error:
        daemon.windows_signal_tree(
            4321,
            9,
            run=lambda *args, **kwargs: result,
            process_exists=lambda pid: True,
        )
    assert error.value.cmd == ["taskkill", "/PID", "4321", "/T", "/F"]
    assert error.value.returncode == 128


@pytest.mark.parametrize("sig", [signal.SIGTERM, 9])
def test_failed_tree_signal_after_process_vanished_is_success(sig):
    result = SimpleNamespace(returncode=128, stdout="not found", stderr="")
    daemon.windows_signal_tree(
        4321,
        sig,
        run=lambda *args, **kwargs: result,
        process_exists=lambda pid: False,
    )


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


def test_windows_signal_compat_adds_missing_sigkill_marker():
    fake = SimpleNamespace(SIGTERM=15)
    result = daemon.ensure_windows_signal_compat(platform="win32", signal_module=fake)
    assert result == 9
    assert fake.SIGKILL == 9


def test_non_windows_signal_compat_does_not_mutate_signal_module():
    fake = SimpleNamespace(SIGTERM=15)
    before = vars(fake).copy()
    assert daemon.ensure_windows_signal_compat(platform="linux", signal_module=fake) == 9
    assert vars(fake) == before


def test_missing_daemon_symbols_fail_clearly():
    with pytest.raises(daemon.QCCDaemonCompatibilityError, match="QCC FreeToken daemon compatibility error"):
        daemon.install_qcc_serve_command_hook("model", SimpleNamespace())
    with pytest.raises(daemon.QCCDaemonCompatibilityError, match="QCC FreeToken daemon compatibility error"):
        daemon.install_windows_process_tree_hook(platform="win32", osproc_module=SimpleNamespace())
    with pytest.raises(daemon.QCCDaemonCompatibilityError, match="QCC FreeToken daemon compatibility error"):
        daemon._daemon_main(SimpleNamespace(), None)
