"""Auto-learner: analyse response signals and update keyword weights."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from .config import Config, get_config, save_config

logger = logging.getLogger("llm-router.learner")

_learn_lock = asyncio.Lock()


def _get_learn_config() -> dict[str, Any] | None:
    """Return learning config dict, or None if disabled."""
    cfg = get_config()
    if not cfg.learning.enabled:
        return None
    return {
        "alpha": cfg.learning.alpha,
        "min_weight": cfg.learning.min_weight,
        "max_weight": cfg.learning.max_weight,
        "max_keywords_per_update": cfg.learning.max_keywords_per_update,
        "protect_manual": cfg.learning.protect_manual,
    }


def _find_matched_keywords(
    all_text: str,
    protect_manual: bool,
    max_kw: int,
) -> list[str]:
    """Return keywords present in text that are eligible for weight update."""
    cfg = get_config()
    matched: list[str] = []
    for rule in cfg.routing.rules:
        if rule.match.type != "keyword":
            continue
        for kw, w in rule.match.keywords.items():
            if kw in all_text:
                if protect_manual and w.source == "manual":
                    continue
                matched.append(kw)
    return matched[:max_kw]


def _update_weights(
    matched: list[str],
    favour_opus: bool,
    alpha: float,
    min_w: float,
    max_w: float,
    now_str: str,
    cfg: Config | None = None,
) -> None:
    """Apply weight adjustments to matched keywords in-place on the config."""
    if cfg is None:
        cfg = get_config()
    for rule in cfg.routing.rules:
        if rule.match.type != "keyword":
            continue
        for kw in matched:
            if kw not in rule.match.keywords:
                continue
            entry = rule.match.keywords[kw]
            if favour_opus:
                entry.weight_a = _clamp(
                    entry.weight_a * (1 - alpha) + 1.0 * alpha,
                    min_w, max_w,
                )
            else:
                entry.weight_b = _clamp(
                    entry.weight_b * (1 - alpha) + 1.0 * alpha,
                    min_w, max_w,
                )
            entry.source = "learned"


async def learn(
    request_messages: list[dict[str, Any]],
    response_body: dict[str, Any],
    elapsed_s: float,
) -> None:
    """Analyse response and update keyword weights asynchronously."""
    cfg_snapshot = _get_learn_config()
    if cfg_snapshot is None:
        return

    all_text = _extract_all_text(request_messages)
    matched = _find_matched_keywords(
        all_text, cfg_snapshot["protect_manual"], cfg_snapshot["max_keywords_per_update"],
    )
    if not matched:
        return

    favour_opus = _should_favour_opus(response_body, elapsed_s)

    async with _learn_lock:
        cfg = get_config()
        now_str = datetime.now(timezone.utc).isoformat()
        _update_weights(
            matched, favour_opus,
            cfg_snapshot["alpha"], cfg_snapshot["min_weight"], cfg_snapshot["max_weight"], now_str,
            cfg,
        )
        await asyncio.to_thread(save_config, cfg)
    logger.debug("learned from %d keywords, favour_opus=%s", len(matched), favour_opus)


def _should_favour_opus(body: dict[str, Any], elapsed_s: float) -> bool:
    """Determine if the response characteristics suggest Opus is more appropriate."""
    usage: dict[str, Any] = body.get("usage", {})
    content: list[dict[str, Any]] = body.get("content", [])
    predicates: list[bool] = [
        int(usage.get("output_tokens", 0)) > 3000,
        sum(1 for b in content if b.get("type") == "tool_use") > 2,
        any(b.get("type") == "thinking" for b in content),
        elapsed_s > 30,
    ]
    return any(predicates)


def _extract_all_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for m in messages:
        content: str | list[dict[str, Any]] = m.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        else:
            for block in content:
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
    return " ".join(parts)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
