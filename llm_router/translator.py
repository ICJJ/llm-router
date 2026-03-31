"""Translate between Anthropic Messages API and OpenAI Chat Completions API."""
from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx

logger = logging.getLogger("llm-router.translator")

_STOP_REASON_MAP: dict[str | None, str] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
}


def _flatten_content(content: str | list[dict[str, Any]]) -> str:
    """Flatten Anthropic content blocks to a plain string."""
    if isinstance(content, str):
        return content
    dropped = [b.get("type") for b in content if b.get("type") != "text"]
    if dropped:
        logger.warning("dropping non-text content blocks during translation: %s", dropped)
    return "".join(
        block.get("text", "")
        for block in content
        if block.get("type") == "text"
    )


def anthropic_to_openai(body: dict[str, Any], target_model: str) -> dict[str, Any]:
    """Convert an Anthropic Messages request body to OpenAI Chat Completions format."""
    oai_messages: list[dict[str, str]] = []

    # System prompt
    system = body.get("system")
    if system:
        sys_text = system if isinstance(system, str) else json.dumps(system)
        oai_messages.append({"role": "system", "content": sys_text})

    # Messages
    for m in body.get("messages", []):
        role = m.get("role", "user")
        content = _flatten_content(m.get("content", ""))
        oai_messages.append({"role": role, "content": content})

    result: dict[str, Any] = {
        "model": target_model,
        "messages": oai_messages,
    }

    # GPT-5.x uses max_completion_tokens instead of max_tokens
    is_gpt5 = target_model.startswith("gpt-5")

    for key in ("temperature", "top_p", "stream"):
        if key in body:
            result[key] = body[key]

    if "max_tokens" in body:
        if is_gpt5:
            result["max_completion_tokens"] = body["max_tokens"]
        else:
            result["max_tokens"] = body["max_tokens"]

    return result


def openai_to_anthropic(resp: dict[str, Any], model_id: str) -> dict[str, Any]:
    """Convert an OpenAI Chat Completions response to Anthropic Messages format."""
    choices = resp.get("choices", [])
    choice: dict[str, Any] = choices[0] if choices else {}
    message: dict[str, Any] = choice.get("message", {})
    content: str = message.get("content", "")
    finish_reason: str | None = choice.get("finish_reason")
    usage = resp.get("usage", {})

    return {
        "id": f"msg_{resp.get('id', 'unknown')}",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": content or ""}],
        "model": model_id,
        "stop_reason": _STOP_REASON_MAP.get(finish_reason, "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


async def translate_openai_stream(
    resp: httpx.Response, model_id: str,
) -> AsyncGenerator[str, None]:
    """Translate an OpenAI SSE stream into Anthropic SSE events."""
    started = False
    finished = False
    output_tokens = 0

    async for raw_line in resp.aiter_lines():
        line = raw_line.strip()
        if not line:
            continue

        if not line.startswith("data: "):
            continue

        payload = line[6:]
        if payload == "[DONE]":
            break

        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue

        choice = data.get("choices", [{}])[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        # First chunk — emit message_start + content_block_start
        if not started:
            started = True
            msg_start: dict[str, Any] = {
                "type": "message_start",
                "message": {
                    "id": f"msg_{data.get('id', 'unknown')}",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model_id,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            }
            yield f"event: message_start\ndata: {json.dumps(msg_start)}\n\n"

            block_start: dict[str, Any] = {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }
            yield f"event: content_block_start\ndata: {json.dumps(block_start)}\n\n"

        # Content delta
        content_piece = delta.get("content")
        if content_piece:
            block_delta: dict[str, Any] = {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": content_piece},
            }
            yield f"event: content_block_delta\ndata: {json.dumps(block_delta)}\n\n"

        # Finish
        if finish_reason:
            finished = True
            usage = data.get("usage", {})
            output_tokens = usage.get("completion_tokens", output_tokens)

            block_stop: dict[str, Any] = {"type": "content_block_stop", "index": 0}
            yield f"event: content_block_stop\ndata: {json.dumps(block_stop)}\n\n"

            msg_delta: dict[str, Any] = {
                "type": "message_delta",
                "delta": {
                    "stop_reason": _STOP_REASON_MAP.get(finish_reason, "end_turn"),
                },
                "usage": {"output_tokens": output_tokens},
            }
            yield f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n"

            msg_stop = {"type": "message_stop"}
            yield f"event: message_stop\ndata: {json.dumps(msg_stop)}\n\n"

    # If stream ended without a finish_reason, send closing events
    if started and not finished:
        block_stop: dict[str, Any] = {"type": "content_block_stop", "index": 0}
        yield f"event: content_block_stop\ndata: {json.dumps(block_stop)}\n\n"

        msg_delta: dict[str, Any] = {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": output_tokens},
        }
        yield f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n"

        msg_stop = {"type": "message_stop"}
        yield f"event: message_stop\ndata: {json.dumps(msg_stop)}\n\n"
