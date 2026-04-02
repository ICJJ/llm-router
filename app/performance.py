"""In-memory performance tracker for model routing."""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class _PerfRecord:
    timestamp: float
    elapsed_ms: float
    output_tokens: int
    success: bool


class PerformanceTracker:
    """Track per-model latency and throughput for performance-based routing."""

    def __init__(self, window_seconds: int = 3600, min_samples: int = 5) -> None:
        self._window = window_seconds
        self._min_samples = min_samples
        self._records: dict[str, deque[_PerfRecord]] = {}
        self._lock = threading.Lock()

    def record(self, model: str, elapsed_ms: float, output_tokens: int, success: bool) -> None:
        now = time.monotonic()
        rec = _PerfRecord(now, elapsed_ms, output_tokens, success)
        with self._lock:
            if model not in self._records:
                self._records[model] = deque()
            q = self._records[model]
            q.append(rec)
            cutoff = now - self._window
            while q and q[0].timestamp < cutoff:
                q.popleft()

    def best_model(self, candidates: list[str], strategy: str = "latency") -> str | None:
        with self._lock:
            scores: dict[str, float] = {}
            for model in candidates:
                q = self._records.get(model)
                if not q:
                    continue
                successes = [r for r in q if r.success]
                if len(successes) < self._min_samples:
                    continue
                if strategy == "latency":
                    scores[model] = sum(r.elapsed_ms for r in successes) / len(successes)
                elif strategy == "throughput":
                    total_tokens = sum(r.output_tokens for r in successes)
                    total_ms = sum(r.elapsed_ms for r in successes)
                    scores[model] = total_tokens / total_ms * 1000 if total_ms > 0 else 0
            if not scores:
                return None
            if strategy == "latency":
                return min(scores, key=scores.get)  # type: ignore[arg-type]
            return max(scores, key=scores.get)  # type: ignore[arg-type]

    def get_stats(self) -> dict[str, dict[str, Any]]:
        """Return per-model stats for health endpoint."""
        with self._lock:
            result: dict[str, dict[str, Any]] = {}
            for model, q in self._records.items():
                successes = [r for r in q if r.success]
                avg_ms = sum(r.elapsed_ms for r in successes) / len(successes) if successes else 0
                result[model] = {
                    "samples": len(q),
                    "successes": len(successes),
                    "failures": len(q) - len(successes),
                    "avg_latency_ms": round(avg_ms),
                }
            return result


# Module-level singleton
_tracker: PerformanceTracker | None = None


def init_tracker(window_seconds: int = 3600, min_samples: int = 5) -> PerformanceTracker:
    global _tracker
    _tracker = PerformanceTracker(window_seconds, min_samples)
    return _tracker


def get_tracker() -> PerformanceTracker:
    global _tracker
    if _tracker is None:
        from .config import get_config
        try:
            cfg = get_config()
            perf = cfg.routing.performance
            _tracker = PerformanceTracker(perf.window_seconds, perf.min_samples)
        except Exception:
            _tracker = PerformanceTracker()
    return _tracker
