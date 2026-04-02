"""Tests for tiered fallback and GPT-5.x parameter handling."""
from __future__ import annotations

from typing import Any

import pytest

from app import fallback
from app.config import Config, CircuitBreakerConfig, FallbackConfig
from app.translator import anthropic_to_openai


# ── Fixtures ────────────────────────────────────────────────────

TIERED_RULES: dict[str, Any] = {
    "default_model": "claude-sonnet-4-6",
    "fallback": {
        "enabled": True,
        "default_tier": "T2",
        "tiers": {
            "T1": ["claude-opus-4-6", "claude-sonnet-4-6", "gpt-5.4", "gpt-4.1", "gpt-4.1-mini"],
            "T2": ["claude-sonnet-4-6", "gpt-4.1", "gpt-4.1-mini"],
            "T3": ["gpt-4.1-mini", "gpt-4.1-nano"],
        },
        "openai_models": ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-5.4"],
        "deployments": {
            "gpt-4.1": "deploy-gpt-4.1",
            "gpt-4.1-mini": "deploy-gpt-4.1-mini",
            "gpt-4.1-nano": "deploy-gpt-4.1-nano",
            "gpt-5.4": "deploy-gpt-5.4",
        },
        "retry_on_status": [403, 500, 502, 503],
        "max_retries": 2,
    },
    "keywords": {
        "选股": {"opus_weight": 0.9, "sonnet_weight": 0.1, "tier": "T1", "source": "manual"},
        "天气": {"opus_weight": 0.1, "sonnet_weight": 0.9, "tier": "T3", "source": "manual"},
        "监控": {"opus_weight": 0.2, "sonnet_weight": 0.8, "tier": "T2", "source": "manual"},
    },
}


@pytest.fixture(autouse=True)
def _reset_health_state():  # pyright: ignore[reportUnusedFunction]
    """Clear module-level health state between tests."""
    fallback._health.clear()  # pyright: ignore[reportPrivateUsage]
    yield
    fallback._health.clear()  # pyright: ignore[reportPrivateUsage]


@pytest.fixture(autouse=True)
def _mock_settings(monkeypatch: pytest.MonkeyPatch):  # pyright: ignore[reportUnusedFunction]
    """Provide deterministic config for all tests."""
    cfg = Config(
        fallback=FallbackConfig(
            circuit_breaker=CircuitBreakerConfig(
                failure_threshold=3,
                recovery_seconds=300,
            ),
        ),
    )
    monkeypatch.setattr("app.fallback.get_config", lambda: cfg)


def _make_all_unhealthy(models: list[str]) -> None:
    """Drive models past the failure threshold."""
    for m in models:
        for _ in range(3):
            fallback.record_failure(m)


# ── get_fallback_model (tiered) ─────────────────────────────────


class TestGetFallbackModelTiered:
    """Tier-aware fallback chain selection."""

    def test_t1_opus_falls_to_sonnet(self):
        model = fallback.get_fallback_model("claude-opus-4-6", TIERED_RULES, tier="T1")
        assert model == "claude-sonnet-4-6"

    def test_t1_sonnet_falls_to_gpt54(self):
        model = fallback.get_fallback_model("claude-sonnet-4-6", TIERED_RULES, tier="T1")
        assert model == "gpt-5.4"

    def test_t1_gpt54_falls_to_gpt41(self):
        model = fallback.get_fallback_model("gpt-5.4", TIERED_RULES, tier="T1")
        assert model == "gpt-4.1"

    def test_t1_gpt41_falls_to_gpt41mini(self):
        model = fallback.get_fallback_model("gpt-4.1", TIERED_RULES, tier="T1")
        assert model == "gpt-4.1-mini"

    def test_t1_full_chain_opus_to_end(self):
        """Walk the full T1 chain by marking each model unhealthy."""
        _make_all_unhealthy(["claude-sonnet-4-6", "gpt-5.4", "gpt-4.1"])
        model = fallback.get_fallback_model("claude-opus-4-6", TIERED_RULES, tier="T1")
        assert model == "gpt-4.1-mini"

    def test_t1_all_exhausted_returns_none(self):
        _make_all_unhealthy(["claude-sonnet-4-6", "gpt-5.4", "gpt-4.1", "gpt-4.1-mini"])
        model = fallback.get_fallback_model("claude-opus-4-6", TIERED_RULES, tier="T1")
        assert model is None

    def test_t2_sonnet_falls_to_gpt41(self):
        model = fallback.get_fallback_model("claude-sonnet-4-6", TIERED_RULES, tier="T2")
        assert model == "gpt-4.1"

    def test_t2_gpt41_falls_to_gpt41mini(self):
        model = fallback.get_fallback_model("gpt-4.1", TIERED_RULES, tier="T2")
        assert model == "gpt-4.1-mini"

    def test_t2_all_exhausted_returns_none(self):
        _make_all_unhealthy(["gpt-4.1", "gpt-4.1-mini"])
        model = fallback.get_fallback_model("claude-sonnet-4-6", TIERED_RULES, tier="T2")
        assert model is None

    def test_t3_gpt41mini_falls_to_gpt41nano(self):
        model = fallback.get_fallback_model("gpt-4.1-mini", TIERED_RULES, tier="T3")
        assert model == "gpt-4.1-nano"

    def test_t3_all_exhausted_returns_none(self):
        _make_all_unhealthy(["gpt-4.1-nano"])
        model = fallback.get_fallback_model("gpt-4.1-mini", TIERED_RULES, tier="T3")
        assert model is None

    def test_model_not_in_tier_starts_from_beginning(self):
        model = fallback.get_fallback_model("unknown-model", TIERED_RULES, tier="T1")
        assert model == "claude-opus-4-6"

    def test_empty_tier_returns_none(self):
        rules: dict[str, Any] = {
            "fallback": {
                "enabled": True,
                "tiers": {"T9": []},
            },
        }
        model = fallback.get_fallback_model("claude-opus-4-6", rules, tier="T9")
        assert model is None

    def test_tier_none_uses_default_tier(self):
        """tier=None should resolve to default_tier (T2)."""
        model = fallback.get_fallback_model("claude-sonnet-4-6", TIERED_RULES, tier=None)
        # T2 chain: sonnet → gpt-4.1 → gpt-4.1-mini
        assert model == "gpt-4.1"

    def test_unknown_tier_falls_back_to_default_tier(self):
        """A tier not in tiers dict → falls to default_tier via legacy path."""
        model = fallback.get_fallback_model("claude-sonnet-4-6", TIERED_RULES, tier="T99")
        # T99 not found → legacy path → models key empty → default_tier T2
        assert model == "gpt-4.1"

    def test_legacy_models_key_backward_compat(self):
        """Rules with flat 'models' key (no tiers) still work."""
        legacy_rules: dict[str, Any] = {
            "fallback": {
                "enabled": True,
                "models": ["claude-opus-4-6", "claude-sonnet-4-6", "gpt-4.1"],
            },
        }
        model = fallback.get_fallback_model("claude-opus-4-6", legacy_rules)
        assert model == "claude-sonnet-4-6"

    def test_t1_skips_unhealthy_midchain(self):
        """Unhealthy model in the middle is skipped."""
        _make_all_unhealthy(["claude-sonnet-4-6"])
        model = fallback.get_fallback_model("claude-opus-4-6", TIERED_RULES, tier="T1")
        assert model == "gpt-5.4"


# ── resolve_tier ────────────────────────────────────────────────


class TestResolveTier:
    """Tier resolution from keyword or model inference."""

    def test_keyword_with_tier_returns_keyword_tier(self):
        tier = fallback.resolve_tier("claude-sonnet-4-6", TIERED_RULES, matched_keyword="选股")
        assert tier == "T1"

    def test_keyword_tier_t3(self):
        tier = fallback.resolve_tier("claude-sonnet-4-6", TIERED_RULES, matched_keyword="天气")
        assert tier == "T3"

    def test_keyword_tier_t2(self):
        tier = fallback.resolve_tier("claude-sonnet-4-6", TIERED_RULES, matched_keyword="监控")
        assert tier == "T2"

    def test_keyword_without_tier_field_returns_default(self):
        rules: dict[str, Any] = {
            **TIERED_RULES,
            "keywords": {
                "测试": {"opus_weight": 0.5, "sonnet_weight": 0.5, "source": "manual"},
            },
        }
        tier = fallback.resolve_tier("claude-sonnet-4-6", rules, matched_keyword="测试")
        assert tier == "T2"

    def test_unknown_keyword_returns_default(self):
        tier = fallback.resolve_tier("claude-sonnet-4-6", TIERED_RULES, matched_keyword="不存在")
        assert tier == "T2"

    def test_no_keyword_opus_infers_t1(self):
        tier = fallback.resolve_tier("claude-opus-4-6", TIERED_RULES, matched_keyword=None)
        assert tier == "T1"

    def test_no_keyword_sonnet_infers_default(self):
        tier = fallback.resolve_tier("claude-sonnet-4-6", TIERED_RULES, matched_keyword=None)
        assert tier == "T2"

    def test_no_keyword_gpt_infers_default(self):
        tier = fallback.resolve_tier("gpt-4.1", TIERED_RULES, matched_keyword=None)
        assert tier == "T2"

    def test_no_fallback_config_opus_returns_t1(self):
        tier = fallback.resolve_tier("claude-opus-4-6", {}, matched_keyword=None)
        assert tier == "T1"

    def test_no_fallback_config_sonnet_returns_t2(self):
        tier = fallback.resolve_tier("claude-sonnet-4-6", {}, matched_keyword=None)
        assert tier == "T2"

    def test_custom_default_tier_respected(self):
        rules: dict[str, Any] = {
            "fallback": {"default_tier": "T3"},
            "keywords": {},
        }
        tier = fallback.resolve_tier("claude-sonnet-4-6", rules, matched_keyword=None)
        assert tier == "T3"


# ── get_tier_chain_length ───────────────────────────────────────


class TestGetTierChainLength:
    def test_t1_length(self):
        assert fallback.get_tier_chain_length("T1", TIERED_RULES) == 5

    def test_t2_length(self):
        assert fallback.get_tier_chain_length("T2", TIERED_RULES) == 3

    def test_t3_length(self):
        assert fallback.get_tier_chain_length("T3", TIERED_RULES) == 2

    def test_unknown_tier_returns_zero(self):
        assert fallback.get_tier_chain_length("T99", TIERED_RULES) == 0

    def test_no_fallback_config_returns_zero(self):
        assert fallback.get_tier_chain_length("T1", {}) == 0


# ── resolve_deployment ──────────────────────────────────────────


class TestResolveDeployment:
    def test_known_model_returns_deployment(self):
        assert fallback.resolve_deployment("gpt-4.1", TIERED_RULES) == "deploy-gpt-4.1"

    def test_gpt41mini_deployment(self):
        assert fallback.resolve_deployment("gpt-4.1-mini", TIERED_RULES) == "deploy-gpt-4.1-mini"

    def test_gpt41nano_deployment(self):
        assert fallback.resolve_deployment("gpt-4.1-nano", TIERED_RULES) == "deploy-gpt-4.1-nano"

    def test_gpt54_deployment(self):
        assert fallback.resolve_deployment("gpt-5.4", TIERED_RULES) == "deploy-gpt-5.4"

    def test_unknown_model_returns_model_name(self):
        assert fallback.resolve_deployment("claude-opus-4-6", TIERED_RULES) == "claude-opus-4-6"

    def test_no_deployments_config_returns_model_name(self):
        assert fallback.resolve_deployment("gpt-4.1", {}) == "gpt-4.1"


# ── is_gpt5_model ──────────────────────────────────────────────


class TestIsGpt5Model:
    def test_gpt54_is_true(self):
        assert fallback.is_gpt5_model("gpt-5.4") is True

    def test_gpt54_mini_is_true(self):
        assert fallback.is_gpt5_model("gpt-5.4-mini") is True

    def test_gpt5_bare_is_true(self):
        assert fallback.is_gpt5_model("gpt-5") is True

    def test_gpt41_is_false(self):
        assert fallback.is_gpt5_model("gpt-4.1") is False

    def test_gpt4o_is_false(self):
        assert fallback.is_gpt5_model("gpt-4o") is False

    def test_claude_is_false(self):
        assert fallback.is_gpt5_model("claude-opus-4-6") is False

    def test_empty_string_is_false(self):
        assert fallback.is_gpt5_model("") is False


# ── anthropic_to_openai GPT-5.x handling ───────────────────────


class TestAnthropicToOpenaiGpt5:
    """GPT-5.x uses max_completion_tokens instead of max_tokens."""

    def test_gpt5_uses_max_completion_tokens(self):
        body: dict[str, Any] = {
            "model": "claude-opus-4-6",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4096,
        }
        result = anthropic_to_openai(body, "gpt-5.4")
        assert "max_completion_tokens" in result
        assert result["max_completion_tokens"] == 4096
        assert "max_tokens" not in result

    def test_gpt5_mini_uses_max_completion_tokens(self):
        body: dict[str, Any] = {
            "model": "claude-opus-4-6",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 2048,
        }
        result = anthropic_to_openai(body, "gpt-5.4-mini")
        assert result["max_completion_tokens"] == 2048
        assert "max_tokens" not in result

    def test_gpt41_uses_max_completion_tokens(self):
        body: dict[str, Any] = {
            "model": "claude-opus-4-6",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4096,
        }
        result = anthropic_to_openai(body, "gpt-4.1")
        assert result["max_completion_tokens"] == 4096
        assert "max_tokens" not in result

    def test_no_max_tokens_neither_key_present(self):
        body: dict[str, Any] = {
            "model": "claude-opus-4-6",
            "messages": [{"role": "user", "content": "hello"}],
        }
        result = anthropic_to_openai(body, "gpt-5.4")
        assert "max_tokens" not in result
        assert "max_completion_tokens" not in result

    def test_no_max_tokens_gpt41_neither_key(self):
        body: dict[str, Any] = {
            "model": "claude-opus-4-6",
            "messages": [{"role": "user", "content": "hello"}],
        }
        result = anthropic_to_openai(body, "gpt-4.1")
        assert "max_tokens" not in result
        assert "max_completion_tokens" not in result

    def test_temperature_and_stream_always_passed(self):
        body: dict[str, Any] = {
            "model": "claude-opus-4-6",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.7,
            "stream": True,
            "max_tokens": 1024,
        }
        result = anthropic_to_openai(body, "gpt-5.4")
        assert result["temperature"] == 0.7
        assert result["stream"] is True
        assert result["max_completion_tokens"] == 1024

    def test_top_p_passed_for_gpt5(self):
        body: dict[str, Any] = {
            "model": "claude-opus-4-6",
            "messages": [{"role": "user", "content": "hello"}],
            "top_p": 0.9,
        }
        result = anthropic_to_openai(body, "gpt-5.4")
        assert result["top_p"] == 0.9
