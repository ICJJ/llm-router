"""Declarative rule engine: evaluate routing rules in priority order."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .config import Config, RoutingRule, MatchRule


@dataclass
class RouteResult:
    model: str
    reason: str
    level: int
    detail: str = ""
    session_key: str = ""
    tier: str | None = None


def route(
    messages: list[dict[str, Any]],
    original_model: str,
    cfg: Config,
) -> RouteResult:
    """Route a request: global_override → rules (first-match) → default."""
    system_text = _extract_system(messages)
    user_text = _extract_last_user(messages)
    all_text = f"{system_text} {user_text}"
    session_key = _extract_session_key(system_text)

    # L0: global override
    if cfg.routing.global_override:
        return RouteResult(
            model=cfg.routing.global_override,
            reason="global_override",
            level=0,
            session_key=session_key,
        )

    # L1-L4: rules (first match wins)
    for rule in cfg.routing.rules:
        result = _evaluate_rule(rule, system_text, user_text, all_text, session_key)
        if result:
            return result

    # L5: default
    return RouteResult(
        model=original_model or cfg.routing.default_model,
        reason="default",
        level=5,
        session_key=session_key,
    )


def _evaluate_rule(
    rule: RoutingRule,
    system_text: str,
    user_text: str,
    all_text: str,
    session_key: str,
) -> RouteResult | None:
    """Evaluate one routing rule against request text."""
    match = rule.match
    text = _get_field_text(match.field, system_text, user_text, all_text)

    if match.type == "pattern":
        return _eval_pattern(rule, match, text, session_key)
    if match.type == "keyword":
        return _eval_keyword(rule, match, text, session_key)
    if match.type == "length":
        return _eval_length(rule, match, text, session_key)
    return None


def _get_field_text(field: str, system_text: str, user_text: str, all_text: str) -> str:
    if field == "system_prompt":
        return system_text
    if field == "user_message":
        return user_text
    return all_text  # "all_text" or anything else


def _eval_pattern(
    rule: RoutingRule, match: MatchRule, text: str, session_key: str,
) -> RouteResult | None:
    target = text
    # If extract regex is set, first extract the key from text
    if match.extract:
        m = re.search(match.extract, text)
        if not m:
            return None
        target = m.group(0)
    # Match pattern against target
    if match.pattern and re.search(re.escape(match.pattern), target):
        return RouteResult(
            model=rule.model,
            reason=f"rule:{rule.name}",
            level=2,
            session_key=session_key,
            tier=rule.tier,
        )
    return None


def _eval_keyword(
    rule: RoutingRule, match: MatchRule, text: str, session_key: str,
) -> RouteResult | None:
    score_a = 0.0
    score_b = 0.0
    matched: list[str] = []
    for kw, w in match.keywords.items():
        if kw in text:
            score_a += w.weight_a
            score_b += w.weight_b
            matched.append(kw)
    if not matched:
        return None
    diff = score_a - score_b
    if abs(diff) <= match.threshold:
        return None
    model = rule.model if diff > 0 else (rule.fallback_model or rule.model)
    detail = f"kw={matched}, a={score_a:.2f}, b={score_b:.2f}"
    return RouteResult(
        model=model,
        reason=f"rule:{rule.name}",
        level=4,
        detail=detail,
        session_key=session_key,
        tier=rule.tier,
    )


def _eval_length(
    rule: RoutingRule, match: MatchRule, text: str, session_key: str,
) -> RouteResult | None:
    length = len(text)
    if match.min_chars and length < match.min_chars:
        return None
    if match.max_chars is not None and length > match.max_chars:
        return None
    return RouteResult(
        model=rule.model,
        reason=f"rule:{rule.name}",
        level=4,
        detail=f"len={length}",
        session_key=session_key,
        tier=rule.tier,
    )


# ── Text extraction helpers (from classifier.py) ──────────────────

def content_to_text(content: str | list[dict[str, Any]]) -> str:
    """Convert Anthropic content (string or content blocks) to plain text."""
    if isinstance(content, str):
        return content
    return " ".join(
        block.get("text", "")
        for block in content
        if block.get("type") == "text"
    )


def _extract_system(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for m in messages:
        if m.get("role") == "system":
            parts.append(content_to_text(m.get("content", "")))
    return " ".join(parts)


def _extract_last_user(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return content_to_text(m.get("content", ""))
    return ""


_SESSION_KEY_RE = re.compile(r"agent:\w+:(?:cron|chat):[^\s]+")


def _extract_session_key(text: str) -> str:
    m = _SESSION_KEY_RE.search(text)
    return m.group(0) if m else ""
