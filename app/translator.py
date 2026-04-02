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

_VERTEX_FINISH_MAP: dict[str | None, str] = {
    "MAX_TOKENS": "length",
    "STOP": "stop",
}


def normalize_amd_response(resp_body: Any) -> dict[str, Any]:
    """Normalize AMD Gateway's non-standard response formats to OpenAI format.

    AMD Gateway returns different formats depending on provider:
    - Azure OpenAI: standard OpenAI format (pass through)
    - Unified endpoint: {"response": {"role": "...", "content": "..."}}
    - VertexAI native: [{"candidates": [...], "usageMetadata": {...}}]
    """
    # Standard OpenAI format — pass through
    if isinstance(resp_body, dict) and "choices" in resp_body:
        return resp_body

    # AMD unified simplified format: {"response": {...}}
    if isinstance(resp_body, dict) and "response" in resp_body:
        r = resp_body["response"]
        role = r.get("role", "assistant")
        if role == "model":
            role = "assistant"
        return {
            "id": f"chatcmpl-amd-{resp_body.get('deployment', 'unknown')}",
            "object": "chat.completion",
            "model": resp_body.get("model", ""),
            "choices": [{
                "index": 0,
                "message": {"role": role, "content": r.get("content", "")},
                "finish_reason": _VERTEX_FINISH_MAP.get(r.get("finishReason"), "stop"),
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    # VertexAI native format: [{"candidates": [...]}]
    if isinstance(resp_body, list) and resp_body:
        item = resp_body[0]
        if isinstance(item, dict) and "candidates" in item:
            candidates = item.get("candidates", [])
            text = ""
            finish = "stop"
            if candidates:
                c = candidates[0]
                parts = c.get("content", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in parts)
                finish = _VERTEX_FINISH_MAP.get(c.get("finishReason"), "stop")
            usage_meta = item.get("usageMetadata", {})
            return {
                "id": f"chatcmpl-vertex-{item.get('responseId', 'unknown')}",
                "object": "chat.completion",
                "model": item.get("modelVersion", ""),
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish,
                }],
                "usage": {
                    "prompt_tokens": usage_meta.get("promptTokenCount", 0),
                    "completion_tokens": usage_meta.get("candidatesTokenCount", 0),
                },
            }

    # Unknown format — return as-is (will likely fail later, but better than swallowing)
    if isinstance(resp_body, dict):
        return resp_body
    return {"choices": [], "usage": {"prompt_tokens": 0, "completion_tokens": 0}}


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

    # Newer OpenAI models use max_completion_tokens instead of max_tokens
    _NEW_API_MODELS = ("gpt-5", "gpt-4.1", "o3", "o4")
    uses_new_api = target_model.startswith(_NEW_API_MODELS)

    for key in ("temperature", "top_p", "stream"):
        if key in body:
            result[key] = body[key]

    # Request usage info in streaming responses
    if body.get("stream"):
        result["stream_options"] = {"include_usage": True}

    if "max_tokens" in body:
        if uses_new_api:
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
    stop_reason = "end_turn"

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
        if not isinstance(data, dict):
            continue

        choices = data.get("choices", [])
        if not choices:
            # Capture usage from stream_options final chunk (empty choices)
            usage = data.get("usage") or {}
            if usage.get("completion_tokens"):
                output_tokens = usage["completion_tokens"]
            continue
        choice = choices[0]
        delta = choice.get("delta") or {}
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

        # Finish — emit content_block_stop, defer message_delta until usage arrives
        if finish_reason:
            finished = True
            stop_reason = _STOP_REASON_MAP.get(finish_reason, "end_turn")
            usage = data.get("usage") or {}
            output_tokens = usage.get("completion_tokens", output_tokens)

            block_stop: dict[str, Any] = {"type": "content_block_stop", "index": 0}
            yield f"event: content_block_stop\ndata: {json.dumps(block_stop)}\n\n"

    # Emit message_delta + message_stop after loop (usage chunk already captured)
    if started:
        # If stream ended without finish_reason, emit missing content_block_stop
        if not finished:
            block_stop = {"type": "content_block_stop", "index": 0}
            yield f"event: content_block_stop\ndata: {json.dumps(block_stop)}\n\n"

        msg_delta: dict[str, Any] = {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason},
            "usage": {"output_tokens": output_tokens},
        }
        yield f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n"

        msg_stop = {"type": "message_stop"}
        yield f"event: message_stop\ndata: {json.dumps(msg_stop)}\n\n"
