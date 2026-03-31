"""Tests for proxy.py bug fixes: _sanitize_error, _record_stats, _process_sse_event, _finalize_stream."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from llm_router import stats
from llm_router.router import RouteResult
from llm_router.proxy import _finalize_stream, _flush_sse_event, _process_sse_event, _record_stats, _sanitize_error  # pyright: ignore[reportPrivateUsage]


@dataclass
class _FakeRoute:
    model: str = "claude-sonnet-4-6"
    reason: str = "default"
    level: int = 0
    detail: str = ""
    session_key: str = "test-session"


@pytest.fixture(autouse=True)
def _init_stores(tmp_path: Path) -> None:  # pyright: ignore[reportUnusedFunction]
    """Initialize stats for each test."""
    spath = tmp_path / "stats.jsonl"
    stats.init(str(spath))


# ── _sanitize_error ──────────────────────────────────────────────


class TestSanitizeError:
    def test_dict_with_error_keeps_type_and_message(self):
        raw = {
            "error": {
                "type": "rate_limit_error",
                "message": "Too many requests",
                "internal_trace_id": "abc123",
                "request_id": "req-xyz",
            }
        }
        result = _sanitize_error(raw, 429)
        assert result == {
            "error": {
                "type": "rate_limit_error",
                "message": "Too many requests",
            }
        }
        # Internal fields must be stripped
        assert "internal_trace_id" not in result["error"]
        assert "request_id" not in result["error"]

    def test_none_returns_generic_error(self):
        result = _sanitize_error(None, 500)
        assert result["error"]["type"] == "api_error"
        assert "500" in result["error"]["message"]

    def test_non_dict_returns_generic_error(self):
        result = _sanitize_error("some string", 502)
        assert result["error"]["type"] == "api_error"
        assert "502" in result["error"]["message"]

    def test_dict_without_error_key_returns_generic(self):
        result = _sanitize_error({"data": "unexpected"}, 503)
        assert result["error"]["type"] == "api_error"
        assert "503" in result["error"]["message"]

    def test_dict_with_non_dict_error_returns_generic(self):
        result = _sanitize_error({"error": "plain string error"}, 400)
        assert result["error"]["type"] == "api_error"
        assert "400" in result["error"]["message"]

    def test_error_missing_type_gets_default(self):
        raw = {"error": {"message": "something broke"}}
        result = _sanitize_error(raw, 500)
        assert result["error"]["type"] == "api_error"
        assert result["error"]["message"] == "something broke"

    def test_error_missing_message_gets_default(self):
        raw = {"error": {"type": "invalid_request_error"}}
        result = _sanitize_error(raw, 422)
        assert result["error"]["type"] == "invalid_request_error"
        assert "422" in result["error"]["message"]


# ── _record_stats ────────────────────────────────────────────────


class TestRecordStats:
    def _read_last_stat(self, tmp_path: Path) -> dict[str, Any]:
        lines = (tmp_path / "stats.jsonl").read_text(encoding="utf-8").strip().splitlines()
        return json.loads(lines[-1])

    def test_actual_model_overrides_route_model(self, tmp_path: Path):
        route = _FakeRoute(model="claude-sonnet-4-6")
        resp_body: dict[str, Any] = {"usage": {"input_tokens": 10, "output_tokens": 5}, "content": []}
        req_body: dict[str, Any] = {"model": "claude-sonnet-4-6"}
        _record_stats(resp_body, route, 0.5, req_body, [], actual_model="gpt-4.1")  # pyright: ignore[reportArgumentType]
        entry = self._read_last_stat(tmp_path)
        assert entry["model_routed"] == "gpt-4.1"

    def test_no_actual_model_falls_back_to_route_model(self, tmp_path: Path):
        route = _FakeRoute(model="claude-sonnet-4-6")
        resp_body: dict[str, Any] = {"usage": {"input_tokens": 10, "output_tokens": 5}, "content": []}
        req_body: dict[str, Any] = {"model": "claude-sonnet-4-6"}
        _record_stats(resp_body, route, 0.5, req_body, [])  # pyright: ignore[reportArgumentType]
        entry = self._read_last_stat(tmp_path)
        assert entry["model_routed"] == "claude-sonnet-4-6"

    def test_fallback_from_field_recorded(self, tmp_path: Path):
        route = _FakeRoute(model="claude-sonnet-4-6")
        resp_body: dict[str, Any] = {"usage": {}, "content": []}
        req_body: dict[str, Any] = {"model": "claude-opus-4-6"}
        _record_stats(resp_body, route, 0.1, req_body, [], fallback_from="claude-opus-4-6", actual_model="gpt-4.1")  # pyright: ignore[reportArgumentType]
        entry = self._read_last_stat(tmp_path)
        assert entry["fallback_from"] == "claude-opus-4-6"
        assert entry["model_routed"] == "gpt-4.1"

    def test_no_fallback_from_when_none(self, tmp_path: Path):
        route = _FakeRoute(model="claude-sonnet-4-6")
        resp_body: dict[str, Any] = {"usage": {}, "content": []}
        req_body: dict[str, Any] = {"model": "claude-sonnet-4-6"}
        _record_stats(resp_body, route, 0.1, req_body, [])  # pyright: ignore[reportArgumentType]
        entry = self._read_last_stat(tmp_path)
        assert "fallback_from" not in entry


# ── _process_sse_event ───────────────────────────────────────────


class TestProcessSseEvent:
    def test_message_delta_uses_state_current_model(self):
        """build_streaming_event should receive state['current_model'], not route.model."""
        route = _FakeRoute(model="claude-sonnet-4-6")
        state: dict[str, Any] = {
            "output_tokens": 10,
            "last_block_index": 0,
            "resp_content_blocks": [],
            "has_text_content": True,
            "injected": False,
            "t0": time.monotonic() - 1.0,
            "current_model": "gpt-4.1",
        }
        inject_cfg: dict[str, Any] = {"enabled": True, "include_model": True}

        data = json.dumps({"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 20}})

        with patch("llm_router.proxy.metadata.build_streaming_event") as mock_build:
            mock_build.return_value = "event: content_block_delta\ndata: {}\n\n"
            _process_sse_event("message_delta", data, state, route, inject_cfg)  # pyright: ignore[reportArgumentType]
            mock_build.assert_called_once()
            # First positional arg should be state["current_model"] = "gpt-4.1"
            call_args = mock_build.call_args
            assert call_args[0][0] == "gpt-4.1"

    def test_message_delta_does_not_use_route_model_after_fallback(self):
        """When current_model differs from route.model, the injected model should be current_model."""
        route = _FakeRoute(model="claude-opus-4-6")
        state: dict[str, Any] = {
            "output_tokens": 0,
            "last_block_index": 0,
            "resp_content_blocks": [],
            "has_text_content": True,
            "injected": False,
            "t0": time.monotonic(),
            "current_model": "claude-sonnet-4-6",
        }
        inject_cfg: dict[str, Any] = {"enabled": True}
        data = json.dumps({"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}})

        with patch("llm_router.proxy.metadata.build_streaming_event") as mock_build:
            mock_build.return_value = None
            _process_sse_event("message_delta", data, state, route, inject_cfg)  # pyright: ignore[reportArgumentType]
            assert mock_build.call_args[0][0] == "claude-sonnet-4-6"


# ── _finalize_stream ─────────────────────────────────────────────


class TestFinalizeStream:
    def test_uses_state_current_model_for_stats(self, tmp_path: Path):
        route = _FakeRoute(model="claude-opus-4-6")
        state: dict[str, Any] = {
            "output_tokens": 42,
            "resp_content_blocks": [],
            "current_model": "gpt-4.1",
        }
        body: dict[str, Any] = {"model": "claude-opus-4-6"}

        with patch("llm_router.proxy._fire_and_forget"):
            _finalize_stream(body, route, time.monotonic() - 1.0, state, [])  # pyright: ignore[reportArgumentType]

        lines = (tmp_path / "stats.jsonl").read_text(encoding="utf-8").strip().splitlines()
        entry = json.loads(lines[-1])
        assert entry["model_routed"] == "gpt-4.1"

    def test_fallback_from_set_when_model_differs(self, tmp_path: Path):
        route = _FakeRoute(model="claude-opus-4-6")
        state: dict[str, Any] = {
            "output_tokens": 10,
            "resp_content_blocks": [],
            "current_model": "gpt-4.1",
        }
        body: dict[str, Any] = {"model": "claude-opus-4-6"}

        with patch("llm_router.proxy._fire_and_forget"):
            _finalize_stream(body, route, time.monotonic(), state, [])  # pyright: ignore[reportArgumentType]

        lines = (tmp_path / "stats.jsonl").read_text(encoding="utf-8").strip().splitlines()
        entry = json.loads(lines[-1])
        assert entry["fallback_from"] == "claude-opus-4-6"

    def test_no_fallback_from_when_model_matches(self, tmp_path: Path):
        route = _FakeRoute(model="claude-sonnet-4-6")
        state: dict[str, Any] = {
            "output_tokens": 10,
            "resp_content_blocks": [],
            "current_model": "claude-sonnet-4-6",
        }
        body: dict[str, Any] = {"model": "claude-sonnet-4-6"}

        with patch("llm_router.proxy._fire_and_forget"):
            _finalize_stream(body, route, time.monotonic(), state, [])  # pyright: ignore[reportArgumentType]

        lines = (tmp_path / "stats.jsonl").read_text(encoding="utf-8").strip().splitlines()
        entry = json.loads(lines[-1])
        assert "fallback_from" not in entry


async def _async_noop() -> None:  # pyright: ignore[reportUnusedFunction]
    """No-op coroutine for mocking learner.learn."""
    pass


# ── SSE buffer boundary tests ───────────────────────────────────


class TestSseEventBoundaries:
    """Every SSE chunk must end with \\n\\n to prevent parser concatenation."""

    def test_content_block_delta_has_proper_terminator(self):
        route = RouteResult(model="test-model", reason="test", level=0)
        state: dict[str, Any] = {
            "output_tokens": 0,
            "last_block_index": 0,
            "resp_content_blocks": [],
            "injected": False,
            "t0": 0.0,
            "current_model": "test-model",
            "reason": "test",
        }
        inject_cfg: dict[str, Any] = {"enabled": True}

        data = json.dumps({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hello"}})
        chunks = _process_sse_event("content_block_delta", data, state, route, inject_cfg)
        for chunk in chunks:
            assert chunk.endswith("\n\n"), f"SSE chunk missing \\n\\n terminator: {chunk!r}"

    def test_content_block_start_has_proper_terminator(self):
        route = RouteResult(model="test-model", reason="test", level=0)
        state: dict[str, Any] = {
            "output_tokens": 0,
            "last_block_index": 0,
            "resp_content_blocks": [],
            "injected": False,
            "t0": 0.0,
            "current_model": "test-model",
            "reason": "test",
        }
        inject_cfg: dict[str, Any] = {"enabled": False}

        data = json.dumps({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
        chunks = _process_sse_event("content_block_start", data, state, route, inject_cfg)
        assert len(chunks) == 1
        assert chunks[0].endswith("\n\n"), f"SSE chunk missing \\n\\n terminator: {chunks[0]!r}"

    def test_metadata_injection_boundaries(self):
        """Metadata injection event must have proper \\n\\n and not merge with next event."""
        route = RouteResult(model="test-model", reason="test", level=0)
        state: dict[str, Any] = {
            "output_tokens": 100,
            "last_block_index": 0,
            "resp_content_blocks": [],
            "has_text_content": True,
            "injected": False,
            "t0": time.monotonic() - 1.0,
            "current_model": "test-model",
            "reason": "test",
        }
        inject_cfg: dict[str, Any] = {"enabled": True}

        data = json.dumps({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 100}})
        chunks = _process_sse_event("message_delta", data, state, route, inject_cfg)
        # Should have 2 chunks: metadata event + original event
        assert len(chunks) == 2
        for chunk in chunks:
            assert chunk.endswith("\n\n"), f"SSE chunk missing \\n\\n terminator: {chunk!r}"

    def test_sequential_events_maintain_boundaries(self):
        """Multiple events processed sequentially must each have proper \\n\\n."""
        route = RouteResult(model="test-model", reason="test", level=0)
        state: dict[str, Any] = {
            "output_tokens": 0,
            "last_block_index": 0,
            "resp_content_blocks": [],
            "injected": False,
            "t0": time.monotonic(),
            "current_model": "test-model",
            "reason": "test",
        }
        inject_cfg: dict[str, Any] = {"enabled": False}

        events = [
            ("content_block_start", json.dumps({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})),
            ("content_block_delta", json.dumps({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}})),
            ("content_block_delta", json.dumps({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " there"}})),
        ]
        all_chunks: list[str] = []
        for event_type, data in events:
            all_chunks.extend(_process_sse_event(event_type, data, state, route, inject_cfg))

        assert len(all_chunks) == 3
        for i, chunk in enumerate(all_chunks):
            assert chunk.endswith("\n\n"), f"Chunk {i} missing \\n\\n terminator: {chunk!r}"

    def test_flush_sse_event_has_proper_terminator(self):
        """_flush_sse_event must also produce properly terminated SSE events."""
        route = RouteResult(model="test-model", reason="test", level=0)
        state: dict[str, Any] = {
            "output_tokens": 0,
            "last_block_index": 0,
            "resp_content_blocks": [],
            "injected": False,
            "t0": time.monotonic(),
            "current_model": "test-model",
            "reason": "test",
        }
        inject_cfg: dict[str, Any] = {"enabled": False}

        data_lines = ['{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}}']
        chunks = _flush_sse_event("content_block_delta", data_lines, state, route, inject_cfg)
        for chunk in chunks:
            assert chunk.endswith("\n\n"), f"Flushed SSE chunk missing \\n\\n terminator: {chunk!r}"

    def test_invalid_json_still_has_proper_terminator(self):
        """Even invalid JSON data should produce a properly terminated SSE event."""
        route = RouteResult(model="test-model", reason="test", level=0)
        state: dict[str, Any] = {
            "output_tokens": 0,
            "last_block_index": 0,
            "resp_content_blocks": [],
            "injected": False,
            "t0": 0.0,
            "current_model": "test-model",
            "reason": "test",
        }
        inject_cfg: dict[str, Any] = {"enabled": False}

        chunks = _process_sse_event("content_block_delta", "not-valid-json", state, route, inject_cfg)
        assert len(chunks) == 1
        assert chunks[0].endswith("\n\n"), f"SSE chunk missing \\n\\n terminator: {chunks[0]!r}"
