"""Torch-free launcher that adds QCC routing to the stock FreeToken daemon."""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import signal
import subprocess
import sys
from pathlib import Path


QCC_LAUNCH_MODULE = "integration.freetoken_qcc.launch"
QCC_MODEL_ENV = "QCC_FT_DAEMON_MODEL"


class QCCDaemonCompatibilityError(RuntimeError):
    pass


def _compatibility_error(message: str) -> QCCDaemonCompatibilityError:
    return QCCDaemonCompatibilityError(f"QCC FreeToken daemon compatibility error: {message}")


def required_daemon_model(environ=None) -> str:
    environ = os.environ if environ is None else environ
    model = environ.get(QCC_MODEL_ENV, "").strip()
    if not model:
        raise _compatibility_error(f"{QCC_MODEL_ENV} is required")
    return model


def ensure_windows_signal_compat(*, platform=None, signal_module=None) -> int:
    """Provide the daemon's logical forced-stop marker on native Windows."""
    platform = sys.platform if platform is None else platform
    signal_module = signal if signal_module is None else signal_module
    if platform != "win32":
        return int(getattr(signal_module, "SIGKILL", 9))
    if not hasattr(signal_module, "SIGKILL"):
        setattr(signal_module, "SIGKILL", 9)
    return int(signal_module.SIGKILL)


def configure_windows_triton_compiler(*, platform=None, environ=None, find_spec=None) -> str | None:
    """Select bundled TinyCC without importing Triton or changing global user state."""
    platform = sys.platform if platform is None else platform
    environ = os.environ if environ is None else environ
    if platform != "win32":
        return None
    if environ.get("CC"):
        return environ["CC"]
    find_spec = importlib.util.find_spec if find_spec is None else find_spec
    spec = find_spec("triton")
    if spec is None:
        raise _compatibility_error("cannot locate Triton for bundled TinyCC")
    if spec.submodule_search_locations:
        package_dir = Path(next(iter(spec.submodule_search_locations)))
    elif spec.origin:
        package_dir = Path(spec.origin).resolve().parent
    else:
        raise _compatibility_error("Triton package location is unavailable")
    compiler = package_dir / "runtime" / "tcc" / "tcc.exe"
    if not compiler.is_file():
        raise _compatibility_error(f"bundled Triton TinyCC is missing: {compiler}")
    environ["CC"] = str(compiler)
    print(f"QCC WINDOWS TRITON COMPILER:\n{compiler}", flush=True)
    return str(compiler)


def install_qcc_serve_command_hook(expected_model=None, serve_manager_module=None):
    """Patch only the daemon command builder, delegating all other models."""
    expected_model = expected_model or required_daemon_model()
    module = serve_manager_module or importlib.import_module("freetoken.daemon.serve_manager")
    original = getattr(module, "build_serve_command", None)
    if not callable(original):
        raise _compatibility_error("freetoken.daemon.serve_manager.build_serve_command is unavailable")
    signature = inspect.signature(original)
    required_names = {"model", "port", "args", "python", "log_dir"}
    if not required_names.issubset(signature.parameters):
        missing = sorted(required_names - set(signature.parameters))
        raise _compatibility_error(f"build_serve_command is missing parameters: {missing}")

    def qcc_aware_build_serve_command(*call_args, **call_kwargs):
        try:
            bound = signature.bind(*call_args, **call_kwargs)
        except TypeError as error:
            raise _compatibility_error(f"cannot bind build_serve_command call: {error}") from error
        bound.apply_defaults()
        values = bound.arguments
        model = values["model"]
        if model != expected_model:
            return original(*call_args, **call_kwargs)
        python = values["python"]
        port = values["port"]
        extra_args = values["args"]
        log_dir = values["log_dir"]
        print(
            "QCC FREETOKEN DAEMON:\n"
            "routing model through integration.freetoken_qcc.launch",
            flush=True,
        )
        argv = [
            python,
            "-m",
            QCC_LAUNCH_MODULE,
            "serve",
            "--model",
            model,
            "--port",
            str(port),
            *extra_args,
        ]
        return argv, os.path.join(log_dir, f"serve-{port}.log")

    module.build_serve_command = qcc_aware_build_serve_command
    return original


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def windows_signal_tree(pid: int, sig, *, run=None, process_exists=None):
    """Signal exactly one daemon-owned Windows process tree with taskkill."""
    run = subprocess.run if run is None else run
    process_exists = _process_exists if process_exists is None else process_exists
    kill_signal = getattr(signal, "SIGKILL", 9)
    forced = int(sig) in {int(kill_signal), 9}
    command = ["taskkill", "/PID", str(int(pid)), "/T"]
    if forced:
        command.append("/F")
    completed = run(command, capture_output=True, text=True, check=False)
    if completed.returncode == 0 or not process_exists(int(pid)):
        return None
    raise subprocess.CalledProcessError(
        completed.returncode, command, output=completed.stdout, stderr=completed.stderr
    )


def install_windows_process_tree_hook(*, platform=None, osproc_module=None, run=None, process_exists=None):
    """Patch FreeToken's signal function only on Windows; preserve its stop sequence."""
    platform = sys.platform if platform is None else platform
    module = osproc_module or importlib.import_module("freetoken.daemon.osproc")
    original = getattr(module, "signal_group", None)
    if not callable(original):
        raise _compatibility_error("freetoken.daemon.osproc.signal_group is unavailable")
    if platform != "win32":
        return original

    def signal_group(pid, sig):
        return windows_signal_tree(pid, sig, run=run, process_exists=process_exists)

    module.signal_group = signal_group
    return original


def _daemon_main(server_module, argv):
    daemon_main = getattr(server_module, "main", None)
    if not callable(daemon_main):
        raise _compatibility_error("freetoken.daemon.server.main is unavailable")
    signature = inspect.signature(daemon_main)
    kwargs = {}
    if "prog" in signature.parameters:
        kwargs["prog"] = "python -m integration.freetoken_qcc_daemon"
    if "argv" in signature.parameters:
        kwargs["argv"] = argv
        return daemon_main(**kwargs)
    if argv is None:
        return daemon_main(**kwargs)
    return daemon_main(argv, **kwargs)


def main(argv=None):
    model = required_daemon_model()
    ensure_windows_signal_compat()
    configure_windows_triton_compiler()
    install_qcc_serve_command_hook(model)
    install_windows_process_tree_hook()
    server = importlib.import_module("freetoken.daemon.server")
    if not callable(getattr(server, "main", None)):
        raise _compatibility_error("freetoken.daemon.server.main is unavailable")
    print(
        "QCC FREETOKEN DAEMON ACTIVE:\n"
        f"model={model}\ncontrol=http://127.0.0.1:1900\nserve_port=1919",
        flush=True,
    )
    return _daemon_main(server, argv)


if __name__ == "__main__":
    raise SystemExit(main())
