"""Tests for the circuit breaker and fallback model selection."""
from __future__ import annotations

import time
from typing import Any

import pytest

from llm_router import fallback
from llm_router.config import Settings


@pytest.fixture(autouse=True)
def _reset_health_state():  # pyright: ignore[reportUnusedFunction]
    """Clear module-level health state between tests."""
    fallback._health.clear()  # pyright: ignore[reportPrivateUsage]
    yield
    fallback._health.clear()  # pyright: ignore[reportPrivateUsage]


@pytest.fixture(autouse=True)
def _mock_settings(monkeypatch: pytest.MonkeyPatch):  # pyright: ignore[reportUnusedFunction]
    """Provide deterministic settings for all tests."""
    settings = Settings(
        fallback_failure_threshold=3,
        fallback_recovery_seconds=300,
    )
    monkeypatch.setattr("llm_router.fallback.get_settings", lambda: settings)


RULES_WITH_FALLBACK: dict[str, Any] = {
    "fallback": {
        "enabled": True,
        "models": [
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "gpt-4.1",
            "gpt-4o",
        ],
        "openai_models": ["gpt-4.1", "gpt-4o", "gpt-4.1-mini", "gpt-4o-mini"],
    },
}

RULES_DISABLED: dict[str, Any] = {
    "fallback": {"enabled": False, "models": []},
}


# ── is_healthy ──────────────────────────────────────────────────


class TestIsHealthy:
    def test_new_model_is_healthy(self):
        assert fallback.is_healthy("claude-opus-4-6") is True

    def test_healthy_model_returns_true(self):
        fallback.record_success("claude-opus-4-6")
        assert fallback.is_healthy("claude-opus-4-6") is True

    def test_becomes_unhealthy_after_threshold_failures(self):
        for _ in range(3):
            fallback.record_failure("claude-opus-4-6")
        assert fallback.is_healthy("claude-opus-4-6") is False

    def test_stays_healthy_below_threshold(self):
        fallback.record_failure("claude-opus-4-6")
        fallback.record_failure("claude-opus-4-6")
        assert fallback.is_healthy("claude-opus-4-6") is True

    def test_auto_recovery_after_recovery_seconds(self, monkeypatch: pytest.MonkeyPatch):
        for _ in range(3):
            fallback.record_failure("claude-opus-4-6")
        assert fallback.is_healthy("claude-opus-4-6") is False

        # Simulate time passing beyond recovery window
        state = fallback._get_state("claude-opus-4-6")  # pyright: ignore[reportPrivateUsage]
        state.last_failure_time = time.monotonic() - 301
        assert fallback.is_healthy("claude-opus-4-6") is True
        # Should also reset consecutive_failures
        assert state.consecutive_failures == 0

    def test_no_recovery_before_recovery_seconds(self, monkeypatch: pytest.MonkeyPatch):
        for _ in range(3):
            fallback.record_failure("claude-opus-4-6")
        state = fallback._get_state("claude-opus-4-6")  # pyright: ignore[reportPrivateUsage]
        state.last_failure_time = time.monotonic() - 100  # Only 100s, need 300
        assert fallback.is_healthy("claude-opus-4-6") is False


# ── record_failure / record_success ─────────────────────────────


class TestRecordFailure:
    def test_increments_consecutive_failures(self):
        fallback.record_failure("claude-opus-4-6")
        state = fallback._get_state("claude-opus-4-6")  # pyright: ignore[reportPrivateUsage]
        assert state.consecutive_failures == 1

    def test_multiple_failures_accumulate(self):
        fallback.record_failure("claude-opus-4-6")
        fallback.record_failure("claude-opus-4-6")
        state = fallback._get_state("claude-opus-4-6")  # pyright: ignore[reportPrivateUsage]
        assert state.consecutive_failures == 2

    def test_sets_last_failure_time(self):
        before = time.monotonic()
        fallback.record_failure("claude-opus-4-6")
        after = time.monotonic()
        state = fallback._get_state("claude-opus-4-6")  # pyright: ignore[reportPrivateUsage]
        assert before <= state.last_failure_time <= after

    def test_marks_unhealthy_at_threshold(self):
        for _ in range(3):
            fallback.record_failure("claude-opus-4-6")
        state = fallback._get_state("claude-opus-4-6")  # pyright: ignore[reportPrivateUsage]
        assert state.is_healthy is False


class TestRecordSuccess:
    def test_resets_consecutive_failures(self):
        fallback.record_failure("claude-opus-4-6")
        fallback.record_failure("claude-opus-4-6")
        fallback.record_success("claude-opus-4-6")
        state = fallback._get_state("claude-opus-4-6")  # pyright: ignore[reportPrivateUsage]
        assert state.consecutive_failures == 0

    def test_marks_healthy(self):
        for _ in range(3):
            fallback.record_failure("claude-opus-4-6")
        assert fallback.is_healthy("claude-opus-4-6") is False
        fallback.record_success("claude-opus-4-6")
        assert fallback.is_healthy("claude-opus-4-6") is True


# ── get_fallback_model ──────────────────────────────────────────


class TestGetFallbackModel:
    def test_returns_next_healthy_model(self):
        model = fallback.get_fallback_model("claude-opus-4-6", RULES_WITH_FALLBACK)
        assert model == "claude-sonnet-4-6"

    def test_skips_unhealthy_models(self):
        for _ in range(3):
            fallback.record_failure("claude-sonnet-4-6")
        model = fallback.get_fallback_model("claude-opus-4-6", RULES_WITH_FALLBACK)
        assert model == "gpt-4.1"

    def test_returns_none_when_all_unhealthy(self):
        for m in ["claude-sonnet-4-6", "gpt-4.1", "gpt-4o"]:
            for _ in range(3):
                fallback.record_failure(m)
        model = fallback.get_fallback_model("claude-opus-4-6", RULES_WITH_FALLBACK)
        assert model is None

    def test_wraps_around_model_list(self):
        # Primary is last in list — should wrap to beginning
        # claude-opus-4-6 is unhealthy, so wrap picks claude-sonnet-4-6
        for _ in range(3):
            fallback.record_failure("claude-opus-4-6")
        model = fallback.get_fallback_model("gpt-4o", RULES_WITH_FALLBACK)
        assert model == "claude-sonnet-4-6"

    def test_wraps_around_to_first_healthy(self):
        # All after primary unhealthy, wraps to first healthy before primary
        for _ in range(3):
            fallback.record_failure("claude-sonnet-4-6")
            fallback.record_failure("gpt-4.1")
        # gpt-4o is primary (last), wrap finds claude-opus-4-6 (healthy)
        model = fallback.get_fallback_model("gpt-4o", RULES_WITH_FALLBACK)
        assert model == "claude-opus-4-6"

    def test_returns_none_when_fallback_disabled(self):
        model = fallback.get_fallback_model("claude-opus-4-6", RULES_DISABLED)
        assert model is None

    def test_returns_none_when_no_fallback_config(self):
        model = fallback.get_fallback_model("claude-opus-4-6", {})
        assert model is None

    def test_primary_not_in_list_starts_from_beginning(self):
        model = fallback.get_fallback_model("unknown-model", RULES_WITH_FALLBACK)
        assert model == "claude-opus-4-6"


# ── is_openai_model ─────────────────────────────────────────────


class TestIsOpenaiModel:
    def test_gpt_prefix_is_openai(self):
        assert fallback.is_openai_model("gpt-4.1") is True
        assert fallback.is_openai_model("gpt-4o-mini") is True

    def test_claude_is_not_openai(self):
        assert fallback.is_openai_model("claude-opus-4-6", RULES_WITH_FALLBACK) is False
        assert fallback.is_openai_model("claude-sonnet-4-6", RULES_WITH_FALLBACK) is False

    def test_model_in_openai_models_list(self):
        rules = {
            "fallback": {"openai_models": ["custom-openai-model"]},
        }
        assert fallback.is_openai_model("custom-openai-model", rules) is True

    def test_model_not_in_openai_models_list(self):
        rules = {
            "fallback": {"openai_models": ["gpt-4.1"]},
        }
        assert fallback.is_openai_model("some-other-model", rules) is False


# ── get_health_status ───────────────────────────────────────────


class TestGetHealthStatus:
    def test_empty_when_no_models_tracked(self):
        assert fallback.get_health_status() == {}

    def test_returns_healthy_model_status(self):
        fallback.record_success("claude-opus-4-6")
        status = fallback.get_health_status()
        assert "claude-opus-4-6" in status
        assert status["claude-opus-4-6"]["healthy"] is True
        assert status["claude-opus-4-6"]["consecutive_failures"] == 0

    def test_returns_unhealthy_model_status(self):
        for _ in range(3):
            fallback.record_failure("claude-opus-4-6")
        status = fallback.get_health_status()
        assert status["claude-opus-4-6"]["healthy"] is False
        assert status["claude-opus-4-6"]["consecutive_failures"] == 3

    def test_multiple_models(self):
        fallback.record_success("claude-opus-4-6")
        for _ in range(3):
            fallback.record_failure("gpt-4.1")
        status = fallback.get_health_status()
        assert len(status) == 2
        assert status["claude-opus-4-6"]["healthy"] is True
        assert status["gpt-4.1"]["healthy"] is False

    def test_includes_last_failure_time(self):
        fallback.record_failure("claude-opus-4-6")
        status = fallback.get_health_status()
        assert status["claude-opus-4-6"]["last_failure_time"] > 0
