"""Python-first launcher which installs a spawn-safe FreeToken scheduler target."""
from __future__ import annotations

import importlib
import json
import os

from integration.freetoken_qcc.bridge import FreeTokenQCCConfig
from integration.freetoken_qcc.low_vram.config import LowVRAMConfig
from integration.freetoken_qcc.worker import install_scheduler_target


def main():
    config = FreeTokenQCCConfig.from_env()
    low_vram = LowVRAMConfig.from_env()
    if low_vram.enabled:
        raise RuntimeError(
            "QCC_FT_LOW_VRAM requested, but no adapter for the installed FreeToken loader has been "
            "validated. Run scripts/inspect_freetoken_install.py on the target Windows installation; "
            "refusing to fall back to full CUDA materialization."
        )
    if config.mode != "off":
        install_scheduler_target(config)
        print(
            "QCC FreeToken parent scheduler target installed: "
            + json.dumps(
                {"pid": os.getpid(), "mode": config.mode, "target_layer": config.layer},
                sort_keys=True,
            ),
            flush=True,
        )
    cli = importlib.import_module("freetoken.cli")
    return cli.main()


if __name__ == "__main__":
    main()
