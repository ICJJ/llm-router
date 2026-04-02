"""Metadata formatting and injection for upstream responses."""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from typing import Any

_CST = timezone(timedelta(hours=8))


def format_line(
    model_id: str,
    elapsed_s: float,
    output_tokens: int,
    reason: str | None = None,
    *,
    stop_reason: str | None = None,
    include_model: bool = True,
    include_elapsed: bool = True,
    include_tokens: bool = True,
    include_reason: bool = False,
    include_timestamp: bool = True,
) -> str:
    """Build the metadata footer line."""
    parts: list[str] = []
    if include_timestamp:
        ts = datetime.now(_CST).strftime("%H:%M:%S")
        parts.append(f"🕐 {ts}")
    if include_model:
        parts.append(f"🤖 {model_id}")
    if include_elapsed:
        parts.append(f"⏱️ {elapsed_s:.1f}s")
    if include_tokens:
        parts.append(f"📊 {output_tokens:,} tokens")
    if include_reason and reason:
        parts.append(f"📌 {reason}")
    if stop_reason == "max_tokens":
        parts.append("⚠️ 截断")
    return "\n\n---\n" + " | ".join(parts)


def inject_non_streaming(
    body: dict[str, Any],
    model_id: str,
    elapsed_s: float,
    reason: str | None = None,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Mutate an Anthropic Messages response body to append metadata."""
    cfg = config or {}
    if not cfg.get("enabled", True):
        return body

    # Only inject metadata on final replies, not tool_use
    stop_reason = body.get("stop_reason")
    if stop_reason not in ("end_turn", "max_tokens", None):
        return body

    output_tokens = body.get("usage", {}).get("output_tokens", 0)
    line = format_line(
        model_id,
        elapsed_s,
        output_tokens,
        reason,
        stop_reason=stop_reason,
        include_model=cfg.get("include_model", True),
        include_elapsed=cfg.get("include_elapsed", True),
        include_tokens=cfg.get("include_tokens", True),
        include_reason=cfg.get("include_reason", False),
        include_timestamp=cfg.get("include_timestamp", True),
    )

    content = body.get("content", [])
    if content and content[-1].get("type") == "text":
        content[-1]["text"] += line
    else:
        content.append({"type": "text", "text": line})

    return body


def build_streaming_event(
    model_id: str,
    elapsed_s: float,
    output_tokens: int,
    reason: str | None = None,
    *,
    stop_reason: str | None = None,
    block_index: int = 0,
    config: dict[str, Any] | None = None,
) -> str | None:
    """Build an SSE content_block_delta event carrying metadata text.

    Returns the SSE text (with event: and data: lines) or None if disabled.
    """
    cfg = config or {}
    if not cfg.get("enabled", True):
        return None

    line = format_line(
        model_id,
        elapsed_s,
        output_tokens,
        reason,
        stop_reason=stop_reason,
        include_model=cfg.get("include_model", True),
        include_elapsed=cfg.get("include_elapsed", True),
        include_tokens=cfg.get("include_tokens", True),
        include_reason=cfg.get("include_reason", False),
        include_timestamp=cfg.get("include_timestamp", True),
    )

    data: dict[str, Any] = {
        "type": "content_block_delta",
        "index": block_index,
        "delta": {"type": "text_delta", "text": line},
    }
    return f"event: content_block_delta\ndata: {json.dumps(data)}\n\n"
