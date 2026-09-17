import pytest

from integration.freetoken_qcc.low_vram.config import LowVRAMConfig


def test_low_vram_defaults_disabled():
    assert LowVRAMConfig.from_env({}) == LowVRAMConfig()


def test_low_vram_requires_cache_and_sparse_off(tmp_path):
    with pytest.raises(ValueError, match="STREAM_CACHE"):
        LowVRAMConfig.from_env({"QCC_FT_LOW_VRAM": "1"})
    with pytest.raises(ValueError, match="MODE=off"):
        LowVRAMConfig.from_env({"QCC_FT_LOW_VRAM": "1", "QCC_FT_STREAM_CACHE": str(tmp_path), "QCC_FT_MODE": "shadow"})


def test_low_vram_sync_configuration(tmp_path):
    config = LowVRAMConfig.from_env({"QCC_FT_LOW_VRAM": "1", "QCC_FT_STREAM_CACHE": str(tmp_path), "QCC_FT_STREAM_ASYNC": "0"})
    assert config.enabled and not config.asynchronous and config.slots == 2


def test_async_rejected_for_correctness_phase(tmp_path):
    with pytest.raises(ValueError, match="ASYNC=0"):
        LowVRAMConfig.from_env({"QCC_FT_LOW_VRAM": "1", "QCC_FT_STREAM_CACHE": str(tmp_path), "QCC_FT_STREAM_ASYNC": "1"})
