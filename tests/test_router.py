"""Tests for the declarative rule engine in llm_router.router."""
from __future__ import annotations

from typing import Any

import pytest

from llm_router.config import Config, RoutingConfig, RoutingRule, MatchRule, KeywordWeight
from llm_router.router import route, content_to_text, RouteResult, _extract_session_key  # pyright: ignore[reportPrivateUsage]


def _make_config(**routing_kwargs: Any) -> Config:
    """Helper to build a Config with routing overrides."""
    routing_data: dict[str, Any] = {"default_model": "claude-sonnet-4-6"}
    routing_data.update(routing_kwargs)
    return Config(routing=RoutingConfig(**routing_data))


def _user_msg(text: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": text}]


def _sys_user_msgs(system: str, user: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ── default / global override ───────────────────────────────────


class TestDefaultRoute:
    def test_no_rules_returns_default(self):
        cfg = _make_config()
        result = route(_user_msg("hello"), "claude-sonnet-4-6", cfg)
        assert result.model == "claude-sonnet-4-6"
        assert result.reason == "default"
        assert result.level == 5

    def test_global_override(self):
        cfg = _make_config(global_override="gpt-4.1")
        result = route(_user_msg("hello"), "claude-sonnet-4-6", cfg)
        assert result.model == "gpt-4.1"
        assert result.reason == "global_override"
        assert result.level == 0


# ── pattern rules ────────────────────────────────────────────────


class TestPatternMatch:
    def test_pattern_match_system_prompt(self):
        cfg = _make_config(rules=[
            {
                "name": "cron",
                "match": {"type": "pattern", "field": "system_prompt", "pattern": "agent:main:cron"},
                "model": "claude-opus-4-6",
            },
        ])
        messages = _sys_user_msgs("You are agent:main:cron:test-job", "Run the report")
        result = route(messages, "claude-sonnet-4-6", cfg)
        assert result.model == "claude-opus-4-6"
        assert "cron" in result.reason

    def test_pattern_no_match_falls_through(self):
        cfg = _make_config(rules=[
            {
                "name": "cron",
                "match": {"type": "pattern", "field": "system_prompt", "pattern": "agent:main:cron"},
                "model": "claude-opus-4-6",
            },
        ])
        messages = _sys_user_msgs("You are a general assistant", "hello")
        result = route(messages, "claude-sonnet-4-6", cfg)
        assert result.model == "claude-sonnet-4-6"
        assert result.reason == "default"

    def test_pattern_extract_then_match(self):
        cfg = _make_config(rules=[
            {
                "name": "agent-extract",
                "match": {
                    "type": "pattern",
                    "field": "system_prompt",
                    "extract": r"agent:\w+:\w+:\S+",
                    "pattern": "agent:main:cron",
                },
                "model": "claude-opus-4-6",
            },
        ])
        messages = _sys_user_msgs("You are agent:main:cron:morning some other text", "go")
        result = route(messages, "claude-sonnet-4-6", cfg)
        assert result.model == "claude-opus-4-6"

    def test_pattern_extract_no_extract_match(self):
        cfg = _make_config(rules=[
            {
                "name": "agent-extract",
                "match": {
                    "type": "pattern",
                    "field": "system_prompt",
                    "extract": r"agent:\w+:\w+:\S+",
                    "pattern": "agent:main:cron",
                },
                "model": "claude-opus-4-6",
            },
        ])
        # No "agent:xxx:yyy:zzz" in system prompt
        messages = _sys_user_msgs("You are a helper", "go")
        result = route(messages, "claude-sonnet-4-6", cfg)
        assert result.reason == "default"


# ── keyword rules ────────────────────────────────────────────────


class TestKeywordMatch:
    def test_keyword_favours_a(self):
        cfg = _make_config(rules=[
            {
                "name": "keywords",
                "match": {
                    "type": "keyword",
                    "field": "all_text",
                    "keywords": {"深度分析": {"weight_a": 0.8, "weight_b": 0.2}},
                    "threshold": 0.15,
                },
                "model": "claude-opus-4-6",
                "fallback_model": "claude-sonnet-4-6",
            },
        ])
        result = route(_user_msg("请做深度分析"), "claude-sonnet-4-6", cfg)
        assert result.model == "claude-opus-4-6"
        assert "keywords" in result.reason

    def test_keyword_favours_b(self):
        cfg = _make_config(rules=[
            {
                "name": "keywords",
                "match": {
                    "type": "keyword",
                    "field": "all_text",
                    "keywords": {"简单查询": {"weight_a": 0.2, "weight_b": 0.8}},
                    "threshold": 0.15,
                },
                "model": "claude-opus-4-6",
                "fallback_model": "claude-sonnet-4-6",
            },
        ])
        result = route(_user_msg("简单查询一下"), "claude-sonnet-4-6", cfg)
        assert result.model == "claude-sonnet-4-6"

    def test_keyword_below_threshold_no_match(self):
        cfg = _make_config(rules=[
            {
                "name": "keywords",
                "match": {
                    "type": "keyword",
                    "field": "all_text",
                    "keywords": {"平衡词": {"weight_a": 0.51, "weight_b": 0.49}},
                    "threshold": 0.15,
                },
                "model": "claude-opus-4-6",
                "fallback_model": "claude-sonnet-4-6",
            },
        ])
        result = route(_user_msg("平衡词测试"), "claude-sonnet-4-6", cfg)
        # abs(0.51 - 0.49) = 0.02 <= 0.15 threshold → no match
        assert result.reason == "default"

    def test_keyword_no_match_in_text(self):
        cfg = _make_config(rules=[
            {
                "name": "keywords",
                "match": {
                    "type": "keyword",
                    "field": "all_text",
                    "keywords": {"深度分析": {"weight_a": 0.8, "weight_b": 0.2}},
                    "threshold": 0.15,
                },
                "model": "claude-opus-4-6",
            },
        ])
        result = route(_user_msg("hello world"), "claude-sonnet-4-6", cfg)
        assert result.reason == "default"


# ── length rules ─────────────────────────────────────────────────


class TestLengthMatch:
    def test_length_match(self):
        cfg = _make_config(rules=[
            {
                "name": "long-text",
                "match": {"type": "length", "field": "user_message", "min_chars": 100},
                "model": "claude-opus-4-6",
            },
        ])
        result = route(_user_msg("x" * 150), "claude-sonnet-4-6", cfg)
        assert result.model == "claude-opus-4-6"

    def test_length_too_short(self):
        cfg = _make_config(rules=[
            {
                "name": "long-text",
                "match": {"type": "length", "field": "user_message", "min_chars": 100},
                "model": "claude-opus-4-6",
            },
        ])
        result = route(_user_msg("short"), "claude-sonnet-4-6", cfg)
        assert result.reason == "default"

    def test_length_too_long(self):
        cfg = _make_config(rules=[
            {
                "name": "short-text",
                "match": {"type": "length", "field": "user_message", "min_chars": 0, "max_chars": 50},
                "model": "claude-opus-4-6",
            },
        ])
        result = route(_user_msg("x" * 100), "claude-sonnet-4-6", cfg)
        assert result.reason == "default"

    def test_length_within_range(self):
        cfg = _make_config(rules=[
            {
                "name": "medium-text",
                "match": {"type": "length", "field": "user_message", "min_chars": 10, "max_chars": 100},
                "model": "claude-opus-4-6",
            },
        ])
        result = route(_user_msg("x" * 50), "claude-sonnet-4-6", cfg)
        assert result.model == "claude-opus-4-6"


# ── first-match-wins ─────────────────────────────────────────────


class TestFirstMatchWins:
    def test_first_matching_rule_wins(self):
        cfg = _make_config(rules=[
            {
                "name": "first",
                "match": {"type": "pattern", "field": "all_text", "pattern": "hello"},
                "model": "claude-opus-4-6",
            },
            {
                "name": "second",
                "match": {"type": "pattern", "field": "all_text", "pattern": "hello"},
                "model": "gpt-4.1",
            },
        ])
        result = route(_user_msg("hello world"), "claude-sonnet-4-6", cfg)
        assert result.model == "claude-opus-4-6"
        assert "first" in result.reason


# ── session key / tier ───────────────────────────────────────────


class TestSessionKeyAndTier:
    def test_session_key_extraction(self):
        messages = _sys_user_msgs("You are agent:main:cron:a-stock-premarket", "go")
        cfg = _make_config()
        result = route(messages, "claude-sonnet-4-6", cfg)
        assert result.session_key == "agent:main:cron:a-stock-premarket"

    def test_session_key_empty_when_no_match(self):
        result = route(_user_msg("hello"), "claude-sonnet-4-6", _make_config())
        assert result.session_key == ""

    def test_route_result_tier(self):
        cfg = _make_config(rules=[
            {
                "name": "tiered",
                "match": {"type": "pattern", "field": "all_text", "pattern": "match-me"},
                "model": "claude-opus-4-6",
                "tier": "T1",
            },
        ])
        result = route(_user_msg("match-me please"), "claude-sonnet-4-6", cfg)
        assert result.tier == "T1"

    def test_route_result_tier_none_when_not_set(self):
        cfg = _make_config(rules=[
            {
                "name": "no-tier",
                "match": {"type": "pattern", "field": "all_text", "pattern": "match-me"},
                "model": "claude-opus-4-6",
            },
        ])
        result = route(_user_msg("match-me"), "claude-sonnet-4-6", cfg)
        assert result.tier is None


# ── content_to_text ──────────────────────────────────────────────


class TestContentToText:
    def test_string_input(self):
        assert content_to_text("hello world") == "hello world"

    def test_content_blocks(self):
        blocks = [
            {"type": "text", "text": "hello"},
            {"type": "image", "source": {}},
            {"type": "text", "text": "world"},
        ]
        assert content_to_text(blocks) == "hello world"

    def test_empty_blocks(self):
        assert content_to_text([]) == ""

    def test_no_text_blocks(self):
        blocks = [{"type": "image", "source": {}}]
        assert content_to_text(blocks) == ""


# ── _extract_session_key ─────────────────────────────────────────


class TestExtractSessionKey:
    def test_cron_key(self):
        assert _extract_session_key("agent:main:cron:a-stock-premarket") == "agent:main:cron:a-stock-premarket"

    def test_chat_key(self):
        assert _extract_session_key("prefix agent:invest:chat:user123 suffix") == "agent:invest:chat:user123"

    def test_no_key(self):
        assert _extract_session_key("just a normal system prompt") == ""


# ── RouteResult dataclass ────────────────────────────────────────


class TestRouteResult:
    def test_defaults(self):
        r = RouteResult(model="m", reason="r", level=0)
        assert r.detail == ""
        assert r.session_key == ""
        assert r.tier is None

    def test_with_all_fields(self):
        r = RouteResult(model="m", reason="r", level=2, detail="d", session_key="s", tier="T1")
        assert r.model == "m"
        assert r.tier == "T1"
