"""JSONL stats logger and reader."""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any

_lock = threading.Lock()
_path: str = ""
_max_bytes: int = 10_485_760


def init(path: str, max_bytes: int = 10_485_760) -> None:
    global _path, _max_bytes
    _path = path
    _max_bytes = max_bytes
    os.makedirs(os.path.dirname(path), exist_ok=True)


def append(entry: dict[str, Any]) -> None:
    """Append one JSON line to stats file. Thread-safe. Rotates when file exceeds _max_bytes."""
    if not _path:
        return
    entry.setdefault("ts", datetime.now(timezone.utc).isoformat())
    line = json.dumps(entry, ensure_ascii=False) + "\n"
    with _lock:
        try:
            if os.path.isfile(_path) and os.path.getsize(_path) >= _max_bytes:
                rotated = _path + ".1"
                os.replace(_path, rotated)
        except OSError:
            pass
        with open(_path, "a", encoding="utf-8") as f:
            f.write(line)


def read_recent(n: int = 10) -> list[dict[str, Any]]:
    """Read the last N entries from stats file using tail-seek."""
    if not _path or not os.path.isfile(_path):
        return []
    try:
        file_size = os.path.getsize(_path)
    except OSError:
        return []
    if file_size == 0:
        return []

    # Read from end — estimate ~300 bytes per JSON line
    chunk_size = min(file_size, n * 400)
    entries: list[dict[str, Any]] = []
    with open(_path, "rb") as f:
        f.seek(max(0, file_size - chunk_size))
        if f.tell() > 0:
            f.readline()  # skip partial first line
        for raw_line in f:
            line = raw_line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
    return entries[-n:]
