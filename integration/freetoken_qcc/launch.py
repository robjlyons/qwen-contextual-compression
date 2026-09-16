"""Python-first launcher which patches an installed FreeToken before its normal CLI starts."""
from __future__ import annotations

import importlib

from integration.freetoken_qcc.bridge import FreeTokenQCCConfig, install_patch


def main():
    config = FreeTokenQCCConfig.from_env()
    bridge = install_patch(config)
    cli = importlib.import_module("freetoken.cli")
    try:
        return cli.main()
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
