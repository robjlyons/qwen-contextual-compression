"""Spawn-safe FreeToken scheduler entry point for the QCC model-worker patch."""
from __future__ import annotations

import importlib
import json
import os


class FreeTokenIntegrationVersionError(RuntimeError):
    """Raised when FreeToken no longer exposes the scheduler hook used by QCC."""


def qcc_scheduler_entry(args, ack_queue):
    """Install QCC in a spawned scheduler process, then run FreeToken's scheduler."""
    from integration.freetoken_qcc.bridge import FreeTokenQCCConfig, install_patch

    config = FreeTokenQCCConfig.from_env()
    if config.mode == "off":
        raise RuntimeError("QCC scheduler worker must not be used when QCC_FT_MODE=off")

    bridge = install_patch(config)
    try:
        # Under multiprocessing spawn this is a fresh interpreter, so this
        # import resolves FreeToken's unmodified scheduler. The identity guard
        # turns an unexpected import/start-method change into an explicit error.
        launch_module = importlib.import_module("freetoken.server.launch")
        original_scheduler = getattr(launch_module, "_run_scheduler", None)
        if original_scheduler is None or not callable(original_scheduler):
            raise FreeTokenIntegrationVersionError(
                "FreeToken does not expose callable freetoken.server.launch._run_scheduler"
            )
        if original_scheduler is qcc_scheduler_entry:
            raise FreeTokenIntegrationVersionError(
                "QCC scheduler entry resolved to itself; refusing recursive scheduler launch"
            )
        print(
            "QCC FreeToken scheduler worker patch active: "
            + json.dumps(
                {"pid": os.getpid(), "mode": config.mode, "target_layer": config.layer},
                sort_keys=True,
            ),
            flush=True,
        )
        return original_scheduler(args, ack_queue)
    finally:
        bridge.close()


def install_scheduler_target(config, launch_module=None) -> bool:
    """Replace only the parent process's scheduler spawn target."""
    if config.mode == "off":
        return False
    if launch_module is None:
        launch_module = importlib.import_module("freetoken.server.launch")
    original_scheduler = getattr(launch_module, "_run_scheduler", None)
    if original_scheduler is None or not callable(original_scheduler):
        raise FreeTokenIntegrationVersionError(
            "Incompatible FreeToken: expected callable freetoken.server.launch._run_scheduler"
        )
    if original_scheduler is qcc_scheduler_entry:
        return True
    launch_module._qcc_original_scheduler = original_scheduler
    launch_module._run_scheduler = qcc_scheduler_entry
    return True
