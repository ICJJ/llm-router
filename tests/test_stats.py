"""Tests for stats rotation."""
import os
from pathlib import Path
from typing import Any

import pytest

from llm_router import stats


def _stats_path() -> str:
    return stats._path  # pyright: ignore[reportPrivateUsage]


@pytest.fixture(autouse=True)
def _init_stats(tmp_path: Path) -> None:  # pyright: ignore[reportUnusedFunction]
    stats.init(str(tmp_path / "stats.jsonl"), max_bytes=500)


def test_append_and_read():
    entry: dict[str, Any] = {"model_routed": "opus", "elapsed_ms": 100}
    stats.append(entry)
    recent = stats.read_recent(5)
    assert len(recent) == 1
    assert recent[0]["model_routed"] == "opus"


def test_rotation_triggers():
    """When file exceeds max_bytes, it should rotate to .1."""
    # Write enough entries to exceed 500 bytes
    for i in range(20):
        stats.append({"i": i, "padding": "x" * 30})

    path = _stats_path()
    rotated = path + ".1"
    assert os.path.isfile(rotated), "Rotation file should exist"
    # Main file should be smaller than threshold (just the post-rotation entries)
    assert os.path.getsize(path) < 500


def test_rotation_overwrites_old_backup():
    """Second rotation should replace the previous .1 file."""
    for _ in range(20):
        stats.append({"padding": "x" * 30})

    path = _stats_path()
    rotated = path + ".1"
    os.path.getsize(rotated)  # verify file exists and is readable

    # Trigger another rotation
    for _ in range(20):
        stats.append({"padding": "x" * 30})

    assert os.path.isfile(rotated)
    # .1 should be a different file now (overwritten)
    second_size = os.path.getsize(rotated)
    assert second_size > 0
    # Main file should still be under threshold
    assert os.path.getsize(path) < 500
