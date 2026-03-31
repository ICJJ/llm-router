"""Async proxy: forward requests to upstream, inject metadata, capture for learning."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Coroutine, cast

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import fallback as fallback_module, learner, metadata, stats, translator
from .commands import execute as execute_command, is_route_command
from .config import get_config
from .providers import get_registry
from .router import RouteResult, route as route_request

logger = logging.getLogger("llm-router.proxy")


def _sanitize_error(resp_body: Any, status_code: int) -> dict[str, Any]:
    """Sanitize upstream error response to avoid leaking internal details."""
    if isinstance(resp_body, dict):
        body_dict = cast(dict[str, Any], resp_body)
        error = body_dict.get("error", {})
        if isinstance(error, dict):
            err = cast(dict[str, Any], error)
            return {"error": {"type": err.get("type", "api_error"), "message": err.get("message", f"upstream returned {status_code}")}}
    return {"error": {"type": "api_error", "message": f"upstream returned {status_code}"}}


def _fire_and_forget(coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
    """Create a task with exception logging to avoid silent failures."""
    task = asyncio.create_task(coro)

    def _on_done(t: asyncio.Task[Any]) -> None:
        if t.cancelled():
            return
        exc = t.exception()
        if exc:
            logger.error("background task failed: %s", exc, exc_info=exc)

    task.add_done_callback(_on_done)
    return task

# Reusable async client — created on first call
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))
    return _client


async def close_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


async def handle_messages(request: Request) -> JSONResponse | StreamingResponse:
    """Main proxy handler for POST /v1/messages."""
    raw = await request.body()
    body = json.loads(raw)

    messages = body.get("messages", [])
    system_text = body.get("system", "")  # Top-level system field

    # Prepend system as a pseudo-message for router
    classify_messages = messages
    if system_text:
        sys_content = system_text if isinstance(system_text, str) else json.dumps(system_text)
        classify_messages = [{"role": "system", "content": sys_content}] + messages

    # Level 1: /route command intercept
    if is_route_command(messages):
        result = execute_command(messages)
        return JSONResponse(content=result)

    # Levels 0-4: route
    cfg = get_config()
    original_model = body.get("model", cfg.routing.default_model)
    route = route_request(classify_messages, original_model, cfg)
    body["model"] = route.model
    logger.info("routed to %s (L%d: %s)", route.model, route.level, route.reason)

    is_stream = body.get("stream", False)

    # Preserve anthropic-version and anthropic-beta from original request
    anthropic_version = request.headers.get("anthropic-version", "2023-06-01")
    anthropic_beta = request.headers.get("anthropic-beta")

    meta = cfg.metadata
    inject_cfg: dict[str, Any] = {
        "enabled": meta.enabled,
        "include_timestamp": "timestamp" in meta.fields,
        "include_model": "model" in meta.fields,
        "include_elapsed": "elapsed" in meta.fields,
        "include_tokens": "tokens" in meta.fields,
        "include_reason": "stop_reason" in meta.fields,
    }

    if is_stream:
        return await _stream_proxy(
            body, route, inject_cfg, classify_messages,
            anthropic_version, anthropic_beta,
        )
    else:
        return await _non_stream_proxy(
            body, route, inject_cfg, classify_messages,
            anthropic_version, anthropic_beta,
        )


async def _non_stream_proxy(
    body: dict[str, Any],
    route: RouteResult,
    inject_cfg: dict[str, Any],
    classify_messages: list[dict[str, Any]],
    anthropic_version: str,
    anthropic_beta: str | None,
) -> JSONResponse:
    cfg = get_config()
    retry_statuses = set(cfg.fallback.retry_on_status)

    # Use tier from route result, or infer from model
    if route.tier:
        tier = route.tier
    elif route.model in ("claude-opus-4-6", "gpt-5.4"):
        tier = "T1"
    else:
        tier = cfg.fallback.default_tier

    # Build compat dict for fallback functions
    rules_compat = cfg.model_dump()
    chain_len = fallback_module.get_tier_chain_length(tier, rules_compat)
    max_retries = max(chain_len - 1, 0) if cfg.fallback.enabled else 0

    current_model = route.model
    client = _get_client()

    for attempt in range(1 + max_retries):
        registry = get_registry()
        provider_type = registry.get_provider_type(current_model)
        needs_translation = provider_type in ("openai", "vertex")

        if needs_translation:
            req_body = translator.anthropic_to_openai(body, current_model)
        else:
            req_body: dict[str, Any] = {**body, "model": current_model}

        url = registry.get_request_url(current_model)
        hdrs = registry.get_request_headers(current_model, anthropic_version, anthropic_beta)

        t0 = time.monotonic()
        resp = await client.post(url, json=req_body, headers=hdrs)
        elapsed = time.monotonic() - t0

        if resp.status_code == 200:
            fallback_module.record_success(current_model)
            resp_body = resp.json()

            if needs_translation:
                resp_body = translator.openai_to_anthropic(resp_body, current_model)

            actual_reason = route.reason if attempt == 0 else f"fallback:{route.model}→{current_model}"
            metadata.inject_non_streaming(
                resp_body, current_model, elapsed, actual_reason, config=inject_cfg,
            )

            _record_stats(resp_body, route, elapsed, body, classify_messages, fallback_from=route.model if attempt > 0 else None, actual_model=current_model)
            _fire_and_forget(learner.learn(classify_messages, resp_body, elapsed))
            return JSONResponse(content=resp_body)

        # Failed
        fallback_module.record_failure(current_model)
        logger.warning(
            "model %s returned %d (attempt %d/%d)",
            current_model, resp.status_code, attempt + 1, 1 + max_retries,
        )

        if resp.status_code not in retry_statuses or attempt >= max_retries:
            try:
                error_content = _sanitize_error(resp.json(), resp.status_code)
            except Exception:
                error_content = _sanitize_error(None, resp.status_code)
            return JSONResponse(content=error_content, status_code=resp.status_code)

        # Find next fallback
        next_model = fallback_module.get_fallback_model(current_model, rules_compat, tier=tier)
        if not next_model:
            try:
                error_content = _sanitize_error(resp.json(), resp.status_code)
            except Exception:
                error_content = _sanitize_error(None, resp.status_code)
            return JSONResponse(content=error_content, status_code=resp.status_code)

        logger.info("falling back from %s to %s", current_model, next_model)
        current_model = next_model

    # Should not reach here, but safety net
    return JSONResponse(content={"error": {"message": "exhausted retries"}}, status_code=502)


def _process_sse_event(
    event_type: str,
    data_str: str,
    state: dict[str, Any],
    route: RouteResult,
    inject_cfg: dict[str, Any],
) -> list[str]:
    chunks: list[str] = []
    try:
        data = json.loads(data_str)
    except json.JSONDecodeError:
        return [f"event: {event_type}\ndata: {data_str}\n\n"]

    if event_type == "content_block_start":
        state["last_block_index"] = data.get("index", 0)
        cb = data.get("content_block", {})
        if cb:
            state["resp_content_blocks"].append(cb)

    if event_type == "content_block_delta":
        delta_body = data.get("delta", {})
        if delta_body.get("type") == "text_delta" and delta_body.get("text"):
            state["has_text_content"] = True

    if event_type == "message_delta":
        usage = data.get("usage", {})
        state["output_tokens"] = usage.get("output_tokens", state["output_tokens"])
        # Only inject metadata on final replies (end_turn/max_tokens), not tool_use
        stop_reason = data.get("delta", {}).get("stop_reason")
        if not state["injected"] and stop_reason in ("end_turn", "max_tokens") and state.get("has_text_content"):
            state["injected"] = True
            elapsed = time.monotonic() - state["t0"]
            meta_event = metadata.build_streaming_event(
                state.get("current_model", route.model), elapsed, state["output_tokens"],
                state.get("reason", route.reason), block_index=state["last_block_index"],
                config=inject_cfg,
            )
            if meta_event:
                chunks.append(meta_event)

    chunks.append(f"event: {event_type}\ndata: {data_str}\n\n")
    return chunks


def _parse_sse_line(line: str, event_type: str, data_lines: list[str]) -> tuple[str, bool]:
    if line.startswith("event: "):
        return line[7:], False
    if line.startswith("data: "):
        data_lines.append(line[6:])
        return event_type, False
    return event_type, True


def _format_raw_data(data_lines: list[str]) -> str:
    return "".join(f"data: {dl}\n" for dl in data_lines) + "\n"


def _finalize_stream(
    body: dict[str, Any],
    route: RouteResult,
    t0: float,
    state: dict[str, Any],
    classify_messages: list[dict[str, Any]],
) -> None:
    elapsed = time.monotonic() - t0
    actual_model = state.get("current_model", route.model)
    stats_entry: dict[str, Any] = {
        "model_requested": body.get("model", ""),
        "model_routed": actual_model,
        "route_reason": state.get("reason", route.reason),
        "elapsed_ms": int(elapsed * 1000),
        "output_tokens": state["output_tokens"],
        "route_level": route.level,
        "session_key": route.session_key,
    }
    if actual_model != body.get("model"):
        stats_entry["fallback_from"] = body.get("model", "")
    stats.append(stats_entry)
    pseudo_body: dict[str, Any] = {
        "usage": {"output_tokens": state["output_tokens"]},
        "content": state["resp_content_blocks"],
    }
    _fire_and_forget(learner.learn(classify_messages, pseudo_body, elapsed))


def _flush_sse_event(
    event_type: str,
    data_lines: list[str],
    state: dict[str, Any],
    route: RouteResult,
    inject_cfg: dict[str, Any],
) -> list[str]:
    """Process accumulated SSE event lines and return chunks to yield."""
    if event_type and data_lines:
        return _process_sse_event(
            event_type, "\n".join(data_lines),
            state, route, inject_cfg,
        )
    if data_lines:
        return [_format_raw_data(data_lines)]
    return []


async def _stream_proxy(
    body: dict[str, Any],
    route: RouteResult,
    inject_cfg: dict[str, Any],
    classify_messages: list[dict[str, Any]],
    anthropic_version: str,
    anthropic_beta: str | None,
) -> StreamingResponse:
    async def event_generator():
        cfg = get_config()
        retry_statuses = set(cfg.fallback.retry_on_status)

        # Use tier from route result, or infer from model
        if route.tier:
            tier = route.tier
        elif route.model in ("claude-opus-4-6", "gpt-5.4"):
            tier = "T1"
        else:
            tier = cfg.fallback.default_tier

        # Build compat dict for fallback functions
        rules_compat = cfg.model_dump()
        chain_len = fallback_module.get_tier_chain_length(tier, rules_compat)
        max_retries = max(chain_len - 1, 0) if cfg.fallback.enabled else 0

        client = _get_client()
        t0 = time.monotonic()
        state: dict[str, Any] = {
            "output_tokens": 0,
            "last_block_index": 0,
            "resp_content_blocks": [],
            "has_text_content": False,
            "injected": False,
            "t0": t0,
            "current_model": route.model,
            "reason": route.reason,
        }

        current_model = route.model

        for attempt in range(1 + max_retries):
            registry = get_registry()
            provider_type = registry.get_provider_type(current_model)
            needs_translation = provider_type in ("openai", "vertex")

            if needs_translation:
                req_body = translator.anthropic_to_openai(body, current_model)
            else:
                req_body: dict[str, Any] = {**body, "model": current_model, "stream": True}

            url = registry.get_request_url(current_model)
            hdrs = registry.get_request_headers(current_model, anthropic_version, anthropic_beta)

            async with client.stream("POST", url, json=req_body, headers=hdrs) as resp:
                if resp.status_code != 200:
                    await resp.aread()  # consume response body
                    fallback_module.record_failure(current_model)

                    if resp.status_code in retry_statuses and attempt < max_retries:
                        next_model = fallback_module.get_fallback_model(current_model, rules_compat, tier=tier)
                        if next_model:
                            logger.info("stream falling back from %s to %s", current_model, next_model)
                            current_model = next_model
                            state["current_model"] = current_model
                            state["reason"] = f"fallback:{route.model}→{current_model}"
                            continue

                    yield f"data: {json.dumps(_sanitize_error(None, resp.status_code))}\n\n"
                    return

                fallback_module.record_success(current_model)

                if needs_translation:
                    try:
                        async for chunk in translator.translate_openai_stream(resp, current_model):
                            evt = ""
                            dlines: list[str] = []
                            for sse_line in chunk.split("\n"):
                                stripped = sse_line.strip()
                                if stripped.startswith("event: "):
                                    evt = stripped[7:]
                                elif stripped.startswith("data: "):
                                    dlines.append(stripped[6:])
                            if evt and dlines:
                                for c in _flush_sse_event(evt, dlines, state, route, inject_cfg):
                                    yield c
                            else:
                                yield chunk
                    except httpx.StreamError as exc:
                        logger.error("upstream OpenAI/Vertex stream interrupted: %s", exc)
                        error_event: dict[str, Any] = {
                            "type": "error",
                            "error": {"type": "overloaded_error", "message": "upstream stream interrupted"},
                        }
                        yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
                        return
                else:
                    buffer = ""
                    event_type = ""
                    data_lines: list[str] = []
                    try:
                        async for chunk in resp.aiter_text():
                            buffer += chunk
                            while "\n" in buffer:
                                line, buffer = buffer.split("\n", 1)
                                line = line.rstrip("\r")

                                if line == "":
                                    for c in _flush_sse_event(event_type, data_lines, state, route, inject_cfg):
                                        yield c
                                    event_type = ""
                                    data_lines = []
                                    continue

                                event_type, forward = _parse_sse_line(line, event_type, data_lines)
                                if forward:
                                    yield f"{line}\n"
                    except httpx.StreamError as exc:
                        logger.error("upstream stream interrupted: %s", exc)
                        error_event: dict[str, Any] = {
                            "type": "error",
                            "error": {"type": "overloaded_error", "message": "upstream stream interrupted"},
                        }
                        yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
                        return

            break  # success, exit retry loop

        _finalize_stream(body, route, t0, state, classify_messages)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _record_stats(
    resp_body: dict[str, Any], route: RouteResult, elapsed: float,
    req_body: dict[str, Any], messages: list[dict[str, Any]],
    fallback_from: str | None = None, actual_model: str | None = None,
) -> None:
    usage = resp_body.get("usage", {})
    content: list[dict[str, Any]] = resp_body.get("content", [])
    entry: dict[str, Any] = {
        "model_requested": req_body.get("model", ""),
        "model_routed": actual_model or route.model,
        "route_reason": route.reason,
        "route_level": route.level,
        "elapsed_ms": int(elapsed * 1000),
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "had_tool_use": any(b.get("type") == "tool_use" for b in content),
        "had_thinking": any(b.get("type") == "thinking" for b in content),
        "session_key": route.session_key,
    }
    if fallback_from:
        entry["fallback_from"] = fallback_from
    stats.append(entry)


# ── Vertex AI → OpenAI format conversion ───────────────────────────

_VERTEX_FINISH_REASON_MAP: dict[str, str] = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
}


def _convert_vertex_non_stream(data: dict[str, Any], model_name: str) -> dict[str, Any]:
    """Convert a Vertex AI non-streaming response to OpenAI chat completion format."""
    candidates = data.get("candidates", [])
    choices: list[dict[str, Any]] = []
    for i, c in enumerate(candidates):
        parts = c.get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)
        raw_reason = c.get("finishReason", "STOP")
        choices.append({
            "index": i,
            "message": {"role": "assistant", "content": text},
            "finish_reason": _VERTEX_FINISH_REASON_MAP.get(raw_reason, "stop"),
        })
    usage_meta = data.get("usageMetadata", {})
    usage = {
        "prompt_tokens": usage_meta.get("promptTokenCount", 0),
        "completion_tokens": usage_meta.get("candidatesTokenCount", 0),
        "total_tokens": usage_meta.get("totalTokenCount", 0),
    } if usage_meta else {}
    return {
        "id": data.get("id", f"chatcmpl-{int(time.time())}"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": choices,
        "usage": usage,
    }


def _convert_vertex_sse_line(payload: str, model_name: str) -> str:
    """Convert a single Vertex AI SSE data payload to OpenAI chunk format."""
    data = json.loads(payload)
    candidates = data.get("candidates", [])
    choices: list[dict[str, Any]] = []
    for i, c in enumerate(candidates):
        delta: dict[str, Any] = {}
        parts = c.get("content", {}).get("parts", [])
        if parts:
            delta["content"] = "".join(p.get("text", "") for p in parts)
        if c.get("content", {}).get("role") == "model":
            delta["role"] = "assistant"
        choice: dict[str, Any] = {"index": i, "delta": delta, "finish_reason": None}
        raw_reason = c.get("finishReason")
        if raw_reason:
            choice["finish_reason"] = _VERTEX_FINISH_REASON_MAP.get(raw_reason, "stop")
        choices.append(choice)
    return json.dumps({
        "id": data.get("id", f"chatcmpl-{int(time.time())}"),
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": choices,
    })


# ── OpenAI Chat Completions pass-through proxy ─────────────────────

async def handle_chat_completions(request: Request) -> JSONResponse | StreamingResponse:
    """Proxy handler for POST /v1/chat/completions (OpenAI format pass-through)."""
    raw = await request.body()
    body = json.loads(raw)

    messages = body.get("messages", [])
    cfg = get_config()
    original_model = body.get("model", cfg.routing.default_model)
    route = route_request(messages, original_model, cfg)
    logger.info("chat/completions routed to %s (L%d: %s)", route.model, route.level, route.reason)

    is_stream = body.get("stream", False)

    if is_stream:
        return await _oai_stream_proxy(body, route)
    else:
        return await _oai_non_stream_proxy(body, route)


async def _oai_non_stream_proxy(
    body: dict[str, Any],
    route: RouteResult,
) -> JSONResponse:
    """OpenAI format non-streaming proxy with fallback."""
    cfg = get_config()
    retry_statuses = set(cfg.fallback.retry_on_status)

    if route.tier:
        tier = route.tier
    elif route.model in ("claude-opus-4-6", "gpt-5.4"):
        tier = "T1"
    else:
        tier = cfg.fallback.default_tier

    rules_compat = cfg.model_dump()
    chain_len = fallback_module.get_tier_chain_length(tier, rules_compat)
    max_retries = max(chain_len - 1, 0) if cfg.fallback.enabled else 0

    current_model = route.model
    client = _get_client()

    for attempt in range(1 + max_retries):
        registry = get_registry()
        url = registry.get_request_url(current_model)
        hdrs = _oai_build_headers(current_model)

        req_body = {**body, "model": current_model}
        # GPT-5.x uses max_completion_tokens
        if current_model.startswith("gpt-5") and "max_tokens" in req_body:
            req_body["max_completion_tokens"] = req_body.pop("max_tokens")

        t0 = time.monotonic()
        resp = await client.post(url, json=req_body, headers=hdrs)
        elapsed = time.monotonic() - t0

        if resp.status_code == 200:
            fallback_module.record_success(current_model)
            resp_body = resp.json()
            # Detect Vertex AI format and convert to OpenAI format
            if "candidates" in resp_body:
                resp_body = _convert_vertex_non_stream(resp_body, current_model)
            else:
                # Fix model name in response
                resp_body["model"] = current_model
            # Inject metadata as content suffix
            meta = cfg.metadata
            if meta.enabled:
                _oai_inject_metadata(resp_body, current_model, elapsed, route.reason if attempt == 0 else f"fallback:{route.model}→{current_model}")
            return JSONResponse(content=resp_body)

        fallback_module.record_failure(current_model)
        logger.warning("oai model %s returned %d (attempt %d/%d)", current_model, resp.status_code, attempt + 1, 1 + max_retries)

        if resp.status_code not in retry_statuses or attempt >= max_retries:
            try:
                return JSONResponse(content=resp.json(), status_code=resp.status_code)
            except Exception:
                return JSONResponse(content={"error": {"message": f"upstream returned {resp.status_code}"}}, status_code=resp.status_code)

        next_model = fallback_module.get_fallback_model(current_model, rules_compat, tier=tier)
        if not next_model:
            try:
                return JSONResponse(content=resp.json(), status_code=resp.status_code)
            except Exception:
                return JSONResponse(content={"error": {"message": f"upstream returned {resp.status_code}"}}, status_code=resp.status_code)

        logger.info("oai falling back from %s to %s", current_model, next_model)
        current_model = next_model

    return JSONResponse(content={"error": {"message": "exhausted retries"}}, status_code=502)


async def _oai_stream_proxy(
    body: dict[str, Any],
    route: RouteResult,
) -> StreamingResponse:
    """OpenAI format streaming proxy with fallback."""
    async def event_generator():
        cfg = get_config()
        retry_statuses = set(cfg.fallback.retry_on_status)

        if route.tier:
            tier = route.tier
        elif route.model in ("claude-opus-4-6", "gpt-5.4"):
            tier = "T1"
        else:
            tier = cfg.fallback.default_tier

        rules_compat = cfg.model_dump()
        chain_len = fallback_module.get_tier_chain_length(tier, rules_compat)
        max_retries = max(chain_len - 1, 0) if cfg.fallback.enabled else 0

        client = _get_client()
        current_model = route.model
        t0 = time.monotonic()

        for attempt in range(1 + max_retries):
            registry = get_registry()
            url = registry.get_request_url(current_model)
            hdrs = _oai_build_headers(current_model)

            req_body = {**body, "model": current_model, "stream": True}
            if current_model.startswith("gpt-5") and "max_tokens" in req_body:
                req_body["max_completion_tokens"] = req_body.pop("max_tokens")

            async with client.stream("POST", url, json=req_body, headers=hdrs) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    fallback_module.record_failure(current_model)

                    if resp.status_code in retry_statuses and attempt < max_retries:
                        next_model = fallback_module.get_fallback_model(current_model, rules_compat, tier=tier)
                        if next_model:
                            logger.info("oai stream falling back from %s to %s", current_model, next_model)
                            current_model = next_model
                            continue

                    yield f"data: {json.dumps({'error': {'message': f'upstream returned {resp.status_code}'}})}\n\n"
                    return

                fallback_module.record_success(current_model)

                # Read full body to detect format (SSE vs raw JSON)
                full_body = await resp.aread()
                text = full_body.decode("utf-8", errors="replace").strip()

                if text.startswith("data: ") or "\ndata: " in text:
                    # SSE format — process line by line
                    for raw_line in text.split("\n"):
                        line = raw_line.strip()
                        if not line:
                            continue

                        if line.startswith("data: "):
                            payload = line[6:]
                            if payload == "[DONE]":
                                meta = cfg.metadata
                                if meta.enabled:
                                    elapsed = time.monotonic() - t0
                                    reason = route.reason if attempt == 0 else f"fallback:{route.model}→{current_model}"
                                    wm = metadata.format_line(current_model, elapsed, 0, reason,
                                        include_model=True, include_elapsed=True, include_tokens=False, include_reason=False, include_timestamp=True)
                                    wm_chunk = json.dumps({
                                        "id": "chatcmpl-meta",
                                        "object": "chat.completion.chunk",
                                        "model": current_model,
                                        "choices": [{"index": 0, "delta": {"content": wm}, "finish_reason": None}],
                                    })
                                    yield f"data: {wm_chunk}\n\n"
                                yield "data: [DONE]\n\n"
                                continue

                            try:
                                chunk = json.loads(payload)
                                if "candidates" in chunk:
                                    converted = _convert_vertex_sse_line(payload, current_model)
                                    yield f"data: {converted}\n\n"
                                else:
                                    chunk["model"] = current_model
                                    yield f"data: {json.dumps(chunk)}\n\n"
                            except json.JSONDecodeError:
                                yield f"{line}\n\n"
                        else:
                            yield f"{line}\n"
                else:
                    # Non-SSE format (e.g. Vertex returns JSON array/object instead of SSE)
                    try:
                        parsed = json.loads(text)
                        if isinstance(parsed, list):
                            parsed = parsed[0] if parsed else {}

                        if "candidates" in parsed:
                            oai_resp = _convert_vertex_non_stream(parsed, current_model)
                        else:
                            oai_resp = parsed
                            oai_resp["model"] = current_model

                        content = oai_resp.get("choices", [{}])[0].get("message", {}).get("content", "")
                        chunk_id = oai_resp.get("id", f"chatcmpl-{int(time.time())}")
                        created = oai_resp.get("created", int(time.time()))
                        finish_reason = oai_resp.get("choices", [{}])[0].get("finish_reason", "stop")

                        meta = cfg.metadata
                        if meta.enabled:
                            elapsed = time.monotonic() - t0
                            reason = route.reason if attempt == 0 else f"fallback:{route.model}→{current_model}"
                            wm = metadata.format_line(current_model, elapsed, 0, reason,
                                include_model=True, include_elapsed=True, include_tokens=False, include_reason=False, include_timestamp=True)
                            content += wm

                        yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': current_model, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': content}, 'finish_reason': None}]})}\n\n"
                        yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': current_model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish_reason}]})}\n\n"
                        yield "data: [DONE]\n\n"
                    except (json.JSONDecodeError, KeyError, IndexError) as exc:
                        logger.error("non-SSE response parse error: %s", exc)
                        yield f"data: {json.dumps({'error': {'message': f'non-SSE format parse error: {exc}'}})}\n\n"

            break  # success

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _oai_build_headers(model: str) -> dict[str, str]:
    """Build upstream request headers for OpenAI-format requests."""
    registry = get_registry()
    r = registry.resolve(model)
    hdrs: dict[str, str] = {"Content-Type": "application/json"}
    hdrs.update(r.extra_headers)
    if r.auth.type == "header_key":
        hdrs[r.auth.header] = r.auth.value
    elif r.auth.type == "bearer":
        hdrs["Authorization"] = f"Bearer {r.auth.value}"
    return hdrs


def _oai_inject_metadata(
    body: dict[str, Any],
    model_id: str,
    elapsed_s: float,
    reason: str,
) -> None:
    """Inject metadata into an OpenAI Chat Completions response."""
    output_tokens = body.get("usage", {}).get("completion_tokens", 0)
    line = metadata.format_line(model_id, elapsed_s, output_tokens, reason,
        include_model=True, include_elapsed=True, include_tokens=True, include_reason=False, include_timestamp=True)
    choices = body.get("choices", [])
    if choices:
        msg = choices[0].get("message", {})
        content = msg.get("content", "")
        if content is not None:
            msg["content"] = content + line
