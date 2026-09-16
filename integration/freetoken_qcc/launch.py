"""Python-first launcher which installs a spawn-safe FreeToken scheduler target."""
from __future__ import annotations

import importlib
import json
import os

from integration.freetoken_qcc.bridge import FreeTokenQCCConfig
from integration.freetoken_qcc.worker import install_scheduler_target


def main():
    config = FreeTokenQCCConfig.from_env()
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
