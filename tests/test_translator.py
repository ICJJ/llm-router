"""Tests for Anthropic ↔ OpenAI format translation."""
from __future__ import annotations

import json
from typing import Any, cast
from unittest.mock import patch

import httpx
import pytest

from llm_router.translator import (
    _flatten_content,  # pyright: ignore[reportPrivateUsage]
    anthropic_to_openai,
    openai_to_anthropic,
    translate_openai_stream,
)


# ── _flatten_content ────────────────────────────────────────────


class TestFlattenContent:
    def test_string_passthrough(self):
        assert _flatten_content("hello world") == "hello world"

    def test_single_text_block(self):
        blocks = [{"type": "text", "text": "hello"}]
        assert _flatten_content(blocks) == "hello"

    def test_multiple_text_blocks(self):
        blocks = [
            {"type": "text", "text": "hello "},
            {"type": "text", "text": "world"},
        ]
        assert _flatten_content(blocks) == "hello world"

    def test_non_text_blocks_ignored(self):
        blocks: list[dict[str, Any]] = [
            {"type": "image", "source": {"data": "..."}},
            {"type": "text", "text": "caption"},
        ]
        assert _flatten_content(blocks) == "caption"

    def test_empty_list(self):
        assert _flatten_content([]) == ""

    def test_empty_string(self):
        assert _flatten_content("") == ""


# ── anthropic_to_openai ────────────────────────────────────────


class TestAnthropicToOpenai:
    def test_basic_conversion(self):
        body: dict[str, Any] = {
            "model": "claude-opus-4-6",
            "messages": [
                {"role": "user", "content": "hello"},
            ],
            "max_tokens": 1024,
        }
        result = anthropic_to_openai(body, "gpt-4.1")
        assert result["model"] == "gpt-4.1"
        assert result["messages"] == [{"role": "user", "content": "hello"}]
        assert result["max_tokens"] == 1024

    def test_system_prompt_becomes_system_message(self):
        body: dict[str, Any] = {
            "model": "claude-opus-4-6",
            "system": "You are helpful.",
            "messages": [
                {"role": "user", "content": "hi"},
            ],
        }
        result = anthropic_to_openai(body, "gpt-4.1")
        assert result["messages"][0] == {"role": "system", "content": "You are helpful."}
        assert result["messages"][1] == {"role": "user", "content": "hi"}

    def test_system_prompt_non_string(self):
        body: dict[str, Any] = {
            "model": "claude-opus-4-6",
            "system": [{"type": "text", "text": "Be helpful"}],
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = anthropic_to_openai(body, "gpt-4.1")
        # Non-string system is json.dumps'd
        assert result["messages"][0]["role"] == "system"
        parsed = json.loads(result["messages"][0]["content"])
        assert parsed == [{"type": "text", "text": "Be helpful"}]

    def test_content_blocks_flattened(self):
        body: dict[str, Any] = {
            "model": "claude-opus-4-6",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "part1 "},
                        {"type": "text", "text": "part2"},
                    ],
                },
            ],
        }
        result = anthropic_to_openai(body, "gpt-4.1")
        assert result["messages"][0]["content"] == "part1 part2"

    def test_string_content_passthrough(self):
        body: dict[str, Any] = {
            "model": "claude-opus-4-6",
            "messages": [{"role": "user", "content": "just a string"}],
        }
        result = anthropic_to_openai(body, "gpt-4.1")
        assert result["messages"][0]["content"] == "just a string"

    def test_optional_fields_mapped(self):
        body: dict[str, Any] = {
            "model": "claude-opus-4-6",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": True,
        }
        result = anthropic_to_openai(body, "gpt-4.1")
        assert result["max_tokens"] == 2048
        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9
        assert result["stream"] is True

    def test_no_optional_fields(self):
        body: dict[str, Any] = {
            "model": "claude-opus-4-6",
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = anthropic_to_openai(body, "gpt-4.1")
        assert "max_tokens" not in result
        assert "temperature" not in result
        assert "stream" not in result

    def test_model_replaced(self):
        body: dict[str, Any] = {
            "model": "claude-opus-4-6",
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = anthropic_to_openai(body, "gpt-4o-mini")
        assert result["model"] == "gpt-4o-mini"

    def test_multi_turn_conversation(self):
        body: dict[str, Any] = {
            "model": "claude-opus-4-6",
            "messages": [
                {"role": "user", "content": "question1"},
                {"role": "assistant", "content": "answer1"},
                {"role": "user", "content": "question2"},
            ],
        }
        result = anthropic_to_openai(body, "gpt-4.1")
        assert len(result["messages"]) == 3
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][1]["role"] == "assistant"
        assert result["messages"][2]["role"] == "user"


# ── openai_to_anthropic ────────────────────────────────────────


class TestOpenaiToAnthropic:
    def test_basic_conversion(self):
        resp: dict[str, Any] = {
            "id": "chatcmpl-abc123",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = openai_to_anthropic(resp, "gpt-4.1")
        assert result["id"] == "msg_chatcmpl-abc123"
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["content"] == [{"type": "text", "text": "Hello!"}]
        assert result["model"] == "gpt-4.1"

    def test_stop_reason_mapped_stop(self):
        resp: dict[str, Any] = {
            "id": "x",
            "choices": [{"message": {"content": ""}, "finish_reason": "stop"}],
            "usage": {},
        }
        result = openai_to_anthropic(resp, "gpt-4.1")
        assert result["stop_reason"] == "end_turn"

    def test_stop_reason_mapped_length(self):
        resp: dict[str, Any] = {
            "id": "x",
            "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
            "usage": {},
        }
        result = openai_to_anthropic(resp, "gpt-4.1")
        assert result["stop_reason"] == "max_tokens"

    def test_stop_reason_unknown_defaults_to_end_turn(self):
        resp: dict[str, Any] = {
            "id": "x",
            "choices": [{"message": {"content": ""}, "finish_reason": "content_filter"}],
            "usage": {},
        }
        result = openai_to_anthropic(resp, "gpt-4.1")
        assert result["stop_reason"] == "end_turn"

    def test_usage_fields_renamed(self):
        resp: dict[str, Any] = {
            "id": "x",
            "choices": [{"message": {"content": ""}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 42, "completion_tokens": 17},
        }
        result = openai_to_anthropic(resp, "gpt-4.1")
        assert result["usage"]["input_tokens"] == 42
        assert result["usage"]["output_tokens"] == 17

    def test_usage_defaults_to_zero(self):
        resp: dict[str, Any] = {
            "id": "x",
            "choices": [{"message": {"content": ""}, "finish_reason": "stop"}],
        }
        result = openai_to_anthropic(resp, "gpt-4.1")
        assert result["usage"]["input_tokens"] == 0
        assert result["usage"]["output_tokens"] == 0

    def test_empty_content(self):
        resp: dict[str, Any] = {
            "id": "x",
            "choices": [{"message": {"content": None}, "finish_reason": "stop"}],
            "usage": {},
        }
        result = openai_to_anthropic(resp, "gpt-4.1")
        assert result["content"] == [{"type": "text", "text": ""}]

    def test_stop_sequence_is_none(self):
        resp: dict[str, Any] = {
            "id": "x",
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {},
        }
        result = openai_to_anthropic(resp, "gpt-4.1")
        assert result["stop_sequence"] is None

    def test_empty_choices_handles_gracefully(self):
        """Empty choices should return a valid response with empty content."""
        resp: dict[str, Any] = {"id": "x", "choices": [], "usage": {}}
        result = openai_to_anthropic(resp, "gpt-4.1")
        assert result["content"] == [{"type": "text", "text": ""}]
        assert result["stop_reason"] == "end_turn"


# ── translate_openai_stream ─────────────────────────────────────


class MockResponse:
    """Mock httpx.Response providing aiter_lines()."""

    def __init__(self, lines: list[str]):
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line


def _make_sse_data(
    content: str | None = None,
    finish_reason: str | None = None,
    chunk_id: str = "chatcmpl-test",
    usage: dict[str, int] | None = None,
) -> str:
    """Build an OpenAI SSE data line."""
    delta: dict[str, Any] = {}
    if content is not None:
        delta["content"] = content
    chunk: dict[str, Any] = {
        "id": chunk_id,
        "choices": [{"delta": delta, "finish_reason": finish_reason}],
    }
    if usage:
        chunk["usage"] = usage
    return f"data: {json.dumps(chunk)}"


class TestTranslateOpenaiStream:
    @pytest.mark.asyncio
    async def test_full_stream_sequence(self):
        lines = [
            _make_sse_data(content="Hello"),
            _make_sse_data(content=" world"),
            _make_sse_data(finish_reason="stop", usage={"completion_tokens": 5}),
            "data: [DONE]",
        ]
        resp = MockResponse(lines)
        events: list[str] = []
        async for event in translate_openai_stream(cast(httpx.Response, resp), "gpt-4.1"):
            events.append(event)

        # Should produce: message_start, content_block_start,
        # 2x content_block_delta, content_block_stop, message_delta, message_stop
        assert len(events) == 7
        assert "message_start" in events[0]
        assert "content_block_start" in events[1]
        assert "content_block_delta" in events[2]
        assert "Hello" in events[2]
        assert "content_block_delta" in events[3]
        assert " world" in events[3]
        assert "content_block_stop" in events[4]
        assert "message_delta" in events[5]
        assert "end_turn" in events[5]
        assert "message_stop" in events[6]

    @pytest.mark.asyncio
    async def test_message_start_contains_model_id(self):
        lines = [
            _make_sse_data(content="hi"),
            _make_sse_data(finish_reason="stop"),
            "data: [DONE]",
        ]
        resp = MockResponse(lines)
        events = [e async for e in translate_openai_stream(cast(httpx.Response, resp), "gpt-4.1")]
        msg_start_data = json.loads(events[0].split("data: ", 1)[1].strip())
        assert msg_start_data["message"]["model"] == "gpt-4.1"

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        resp = MockResponse(["data: [DONE]"])
        events = [e async for e in translate_openai_stream(cast(httpx.Response, resp), "gpt-4.1")]
        assert events == []

    @pytest.mark.asyncio
    async def test_blank_lines_skipped(self):
        lines = [
            "",
            _make_sse_data(content="Hello"),
            "",
            _make_sse_data(finish_reason="stop"),
            "data: [DONE]",
        ]
        resp = MockResponse(lines)
        events = [e async for e in translate_openai_stream(cast(httpx.Response, resp), "gpt-4.1")]
        # message_start, content_block_start, delta, block_stop, msg_delta, msg_stop
        assert len(events) == 6

    @pytest.mark.asyncio
    async def test_non_data_lines_skipped(self):
        lines = [
            "event: ping",
            _make_sse_data(content="Hello"),
            _make_sse_data(finish_reason="stop"),
            "data: [DONE]",
        ]
        resp = MockResponse(lines)
        events = [e async for e in translate_openai_stream(cast(httpx.Response, resp), "gpt-4.1")]
        assert len(events) == 6

    @pytest.mark.asyncio
    async def test_invalid_json_skipped(self):
        lines = [
            "data: {invalid json",
            _make_sse_data(content="Hello"),
            _make_sse_data(finish_reason="stop"),
            "data: [DONE]",
        ]
        resp = MockResponse(lines)
        events = [e async for e in translate_openai_stream(cast(httpx.Response, resp), "gpt-4.1")]
        assert len(events) == 6

    @pytest.mark.asyncio
    async def test_finish_reason_length_mapped(self):
        lines = [
            _make_sse_data(content="truncated"),
            _make_sse_data(finish_reason="length"),
            "data: [DONE]",
        ]
        resp = MockResponse(lines)
        events = [e async for e in translate_openai_stream(cast(httpx.Response, resp), "gpt-4.1")]
        # Find the message_delta event
        msg_delta_event = [e for e in events if "message_delta" in e][0]
        data = json.loads(msg_delta_event.split("data: ", 1)[1].strip())
        assert data["delta"]["stop_reason"] == "max_tokens"

    @pytest.mark.asyncio
    async def test_content_delta_text_value(self):
        lines = [
            _make_sse_data(content="exact text"),
            _make_sse_data(finish_reason="stop"),
            "data: [DONE]",
        ]
        resp = MockResponse(lines)
        events = [e async for e in translate_openai_stream(cast(httpx.Response, resp), "gpt-4.1")]
        delta_event = events[2]  # After message_start and content_block_start
        data = json.loads(delta_event.split("data: ", 1)[1].strip())
        assert data["delta"]["text"] == "exact text"

    @pytest.mark.asyncio
    async def test_output_tokens_in_message_delta(self):
        lines = [
            _make_sse_data(content="hi"),
            _make_sse_data(finish_reason="stop", usage={"completion_tokens": 42}),
            "data: [DONE]",
        ]
        resp = MockResponse(lines)
        events = [e async for e in translate_openai_stream(cast(httpx.Response, resp), "gpt-4.1")]
        msg_delta_event = [e for e in events if "message_delta" in e][0]
        data = json.loads(msg_delta_event.split("data: ", 1)[1].strip())
        assert data["usage"]["output_tokens"] == 42

    @pytest.mark.asyncio
    async def test_no_content_delta_for_empty_content(self):
        """Chunks with no content in delta should not produce content_block_delta."""
        lines = [
            _make_sse_data(content=None),  # First chunk, no content
            _make_sse_data(content="hello"),
            _make_sse_data(finish_reason="stop"),
            "data: [DONE]",
        ]
        resp = MockResponse(lines)
        events = [e async for e in translate_openai_stream(cast(httpx.Response, resp), "gpt-4.1")]
        delta_events = [e for e in events if "content_block_delta" in e]
        assert len(delta_events) == 1
        data = json.loads(delta_events[0].split("data: ", 1)[1].strip())
        assert data["delta"]["text"] == "hello"

    # ── Bug fix: stream ends without finish_reason ──────────────

    @pytest.mark.asyncio
    async def test_stream_without_finish_reason_emits_closing_events(self):
        """When stream ends without a finish_reason, closing events must be emitted."""
        lines = [
            _make_sse_data(content="partial"),
            _make_sse_data(content=" response"),
            # No finish_reason chunk, stream just ends
            "data: [DONE]",
        ]
        resp = MockResponse(lines)
        events = [e async for e in translate_openai_stream(cast(httpx.Response, resp), "gpt-4.1")]

        # Should still get: message_start, content_block_start,
        # 2x content_block_delta, content_block_stop, message_delta, message_stop
        assert len(events) == 7

        # Last 3 events should be closing events
        assert "content_block_stop" in events[4]
        assert "message_delta" in events[5]
        assert "message_stop" in events[6]

        # message_delta should have stop_reason = end_turn
        msg_delta_data = json.loads(events[5].split("data: ", 1)[1].strip())
        assert msg_delta_data["delta"]["stop_reason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_stream_with_finish_reason_no_duplicate_closing(self):
        """Normal stream with finish_reason should NOT produce duplicate closing events."""
        lines = [
            _make_sse_data(content="hello"),
            _make_sse_data(finish_reason="stop"),
            "data: [DONE]",
        ]
        resp = MockResponse(lines)
        events = [e async for e in translate_openai_stream(cast(httpx.Response, resp), "gpt-4.1")]

        # Normal: message_start, block_start, delta, block_stop, msg_delta, msg_stop
        assert len(events) == 6
        stop_events = [e for e in events if "content_block_stop" in e]
        assert len(stop_events) == 1
        msg_stop_events = [e for e in events if "message_stop" in e]
        assert len(msg_stop_events) == 1

    @pytest.mark.asyncio
    async def test_stream_without_finish_no_data_no_closing(self):
        """Empty stream (no chunks started) should NOT emit closing events."""
        lines = ["data: [DONE]"]
        resp = MockResponse(lines)
        events = [e async for e in translate_openai_stream(cast(httpx.Response, resp), "gpt-4.1")]
        assert events == []


# ── Bug fix: _flatten_content warns on non-text blocks ──────────


class TestFlattenContentWarning:
    def test_tool_use_block_logs_warning_and_returns_text(self):
        blocks: list[dict[str, Any]] = [
            {"type": "text", "text": "hello"},
            {"type": "tool_use", "id": "t1", "name": "calc", "input": {}},
        ]
        with patch("llm_router.translator.logger") as mock_logger:
            result = _flatten_content(blocks)
            assert result == "hello"
            mock_logger.warning.assert_called_once()
            # Verify logged types include tool_use
            call_args = mock_logger.warning.call_args
            assert "tool_use" in str(call_args)

    def test_thinking_block_logs_warning(self):
        blocks = [
            {"type": "thinking", "thinking": "..."},
            {"type": "text", "text": "result"},
        ]
        with patch("llm_router.translator.logger") as mock_logger:
            result = _flatten_content(blocks)
            assert result == "result"
            mock_logger.warning.assert_called_once()
            assert "thinking" in str(mock_logger.warning.call_args)

    def test_multiple_non_text_blocks_single_warning(self):
        blocks: list[dict[str, Any]] = [
            {"type": "tool_use", "id": "t1", "name": "a", "input": {}},
            {"type": "thinking", "thinking": "..."},
            {"type": "text", "text": "ok"},
        ]
        with patch("llm_router.translator.logger") as mock_logger:
            result = _flatten_content(blocks)
            assert result == "ok"
            mock_logger.warning.assert_called_once()
            # Both types should be in the dropped list
            call_arg_str = str(mock_logger.warning.call_args)
            assert "tool_use" in call_arg_str
            assert "thinking" in call_arg_str

    def test_pure_text_no_warning(self):
        blocks = [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": " world"},
        ]
        with patch("llm_router.translator.logger") as mock_logger:
            result = _flatten_content(blocks)
            assert result == "hello world"
            mock_logger.warning.assert_not_called()

    def test_string_input_no_warning(self):
        with patch("llm_router.translator.logger") as mock_logger:
            result = _flatten_content("just a string")
            assert result == "just a string"
            mock_logger.warning.assert_not_called()
