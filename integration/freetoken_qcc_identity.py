"""Torch-free model identity helpers shared by QCC daemon and workers."""
from __future__ import annotations

import ntpath
import os
import sys
from typing import NamedTuple


QCC_MODEL_ENV = "QCC_FT_DAEMON_MODEL"
QCC_MODEL_PATH_ENV = "QCC_FT_DAEMON_MODEL_PATH"


class QCCModelIdentity(NamedTuple):
    launch_model: str | None
    cache_source_model: str | None
    alias: str


def normalized_model_path(path: str, *, platform: str | None = None) -> str:
    """Normalize an exact model path using the target platform's semantics."""
    platform = sys.platform if platform is None else platform
    path_module = ntpath if platform == "win32" else os.path
    return path_module.normcase(path_module.abspath(path))


def paths_equivalent(left: str, right: str, *, platform: str | None = None) -> bool:
    return normalized_model_path(left, platform=platform) == normalized_model_path(
        right, platform=platform
    )


def is_qcc_model(
    model: str,
    expected_model: str,
    configured_local_path: str | None = None,
    *,
    platform: str | None = None,
) -> bool:
    """Match only the configured Hub ID or exact configured local model path."""
    if model == expected_model:
        return True
    return bool(
        configured_local_path
        and paths_equivalent(model, configured_local_path, platform=platform)
    )


def resolve_cache_model_identity(
    launch_model: str | None,
    *,
    configured_model: str | None = None,
    configured_local_path: str | None = None,
    platform: str | None = None,
    environ=None,
) -> QCCModelIdentity:
    """Resolve a launch location to the exact source identity stored in a cache."""
    environ = os.environ if environ is None else environ
    if configured_model is None:
        configured_model = environ.get(QCC_MODEL_ENV, "").strip() or None
    if configured_local_path is None:
        configured_local_path = environ.get(QCC_MODEL_PATH_ENV, "").strip() or None

    if configured_model and launch_model == configured_model:
        return QCCModelIdentity(launch_model, configured_model, "canonical")
    if (
        launch_model
        and configured_model
        and configured_local_path
        and paths_equivalent(launch_model, configured_local_path, platform=platform)
    ):
        return QCCModelIdentity(launch_model, configured_model, "local-configured-path")
    return QCCModelIdentity(launch_model, launch_model, "launch-model")
