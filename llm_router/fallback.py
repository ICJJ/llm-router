"""Circuit breaker and fallback model selection."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .config import get_settings


@dataclass
class _ModelState:
    consecutive_failures: int = 0
    last_failure_time: float = 0.0
    is_healthy: bool = True


_health: dict[str, _ModelState] = {}


def _get_state(model: str) -> _ModelState:
    if model not in _health:
        _health[model] = _ModelState()
    return _health[model]


def is_healthy(model: str) -> bool:
    """Check if a model is healthy, auto-recovering after recovery_seconds."""
    state = _get_state(model)
    if state.is_healthy:
        return True
    settings = get_settings()
    elapsed = time.monotonic() - state.last_failure_time
    if elapsed >= settings.fallback_recovery_seconds:
        state.is_healthy = True
        state.consecutive_failures = 0
        return True
    return False


def record_failure(model: str) -> None:
    """Record a failure for the given model."""
    state = _get_state(model)
    state.consecutive_failures += 1
    state.last_failure_time = time.monotonic()
    settings = get_settings()
    if state.consecutive_failures >= settings.fallback_failure_threshold:
        state.is_healthy = False


def record_success(model: str) -> None:
    """Record a success, resetting failure counters."""
    state = _get_state(model)
    state.consecutive_failures = 0
    state.is_healthy = True


def get_fallback_model(primary: str, rules: dict[str, Any], tier: str | None = None) -> str | None:
    """Find the next healthy model from the tier chain after *primary*."""
    fallback_cfg = rules.get("fallback", {})
    if not fallback_cfg.get("enabled"):
        return None

    # Get the tier chain
    tiers = fallback_cfg.get("tiers", {})
    if tier and tier in tiers:
        models = tiers[tier]
    else:
        # Legacy flat fallback support
        models = fallback_cfg.get("models", [])
        if not models:
            default_tier = fallback_cfg.get("default_tier", "T2")
            models = tiers.get(default_tier, [])

    if primary in models:
        start = models.index(primary) + 1
    else:
        start = 0
    for model in models[start:]:
        if model != primary and is_healthy(model):
            return model
    # Wrap around to the beginning
    for model in models[:start]:
        if model != primary and is_healthy(model):
            return model
    return None


def resolve_tier(route_model: str, rules: dict[str, Any], matched_keyword: str | None = None) -> str:
    """Resolve which tier to use based on matched keyword or infer from model."""
    fallback_cfg = rules.get("fallback", {})
    default_tier = fallback_cfg.get("default_tier", "T2")

    if matched_keyword:
        keywords = rules.get("keywords", {})
        kw_cfg = keywords.get(matched_keyword, {})
        tier = kw_cfg.get("tier")
        if tier:
            return tier

    # Infer tier from route model when keyword tier not available:
    # top-tier models → critical tasks → T1, others → default
    if route_model in ("claude-opus-4-6", "gpt-5.4"):
        return "T1"

    return default_tier


def get_tier_chain_length(tier: str, rules: dict[str, Any]) -> int:
    """Return the number of models in a tier chain (for dynamic max_retries)."""
    fallback_cfg = rules.get("fallback", {})
    tiers = fallback_cfg.get("tiers", {})
    return len(tiers.get(tier, []))


def resolve_deployment(model: str, rules: dict[str, Any]) -> str:
    """Resolve Azure OpenAI deployment name for a model."""
    deployments = rules.get("fallback", {}).get("deployments", {})
    return deployments.get(model, model)


def is_gpt5_model(model: str) -> bool:
    """Check if model is GPT-5.x (requires different API parameters)."""
    return model.startswith("gpt-5")


def is_openai_model(model: str, rules: dict[str, Any] | None = None) -> bool:
    """Check whether *model* is an OpenAI model."""
    if model.startswith("gpt-"):
        return True
    if rules is None:
        from .config import get_config
        cfg = get_config()
        openai_models = {m for _, p in cfg.providers.items()
                         if p.type == "openai" for m in p.models}
        return model in openai_models
    openai_models = rules.get("fallback", {}).get("openai_models", [])
    return model in openai_models


def is_vertex_model(model: str, rules: dict[str, Any] | None = None) -> bool:
    """Check whether *model* is a Vertex AI model (e.g. Gemini)."""
    if model.startswith("gemini-"):
        return True
    if rules is None:
        from .config import get_config
        cfg = get_config()
        vertex_models = {m for _, p in cfg.providers.items()
                         if p.type == "vertex" for m in p.models}
        return model in vertex_models
    vertex_models = rules.get("fallback", {}).get("vertex_models", [])
    return model in vertex_models


def get_health_status() -> dict[str, Any]:
    """Return health status for all tracked models."""
    result: dict[str, Any] = {}
    for model, state in _health.items():
        # Re-evaluate health (may auto-recover)
        healthy = is_healthy(model)
        result[model] = {
            "healthy": healthy,
            "consecutive_failures": state.consecutive_failures,
            "last_failure_time": state.last_failure_time,
        }
    return result
