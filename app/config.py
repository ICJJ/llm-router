"""YAML-based configuration with Pydantic v2 validation and hot-reload."""
from __future__ import annotations

import logging
import os
import re
import sys
import threading
from functools import lru_cache
from io import IOBase
from pathlib import Path
from typing import Any, Literal

_logger = logging.getLogger("llm-router.config")

from pydantic import BaseModel
from pydantic_settings import BaseSettings


# ── Cross-platform file locking (from rules_store.py) ──────────────
if sys.platform == "win32":
    import msvcrt

    def _lock_shared(f: IOBase) -> None:
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)

    def _lock_exclusive(f: IOBase) -> None:
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)

    def _unlock(f: IOBase) -> None:
        try:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
else:
    import fcntl

    def _lock_shared(f: IOBase) -> None:
        fcntl.flock(f, fcntl.LOCK_SH)

    def _lock_exclusive(f: IOBase) -> None:
        fcntl.flock(f, fcntl.LOCK_EX)

    def _unlock(f: IOBase) -> None:
        fcntl.flock(f, fcntl.LOCK_UN)


# ── Pydantic v2 Config Models ──────────────────────────────────────

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8100
    log_level: Literal["debug", "info", "warning", "error"] = "warning"


class AuthConfig(BaseModel):
    type: Literal["bearer", "header_key", "none"] = "bearer"
    header: str = "Authorization"
    value: str = ""


class ProviderConfig(BaseModel):
    type: Literal["anthropic", "openai", "vertex"]
    base_url: str
    auth: AuthConfig = AuthConfig()
    headers: dict[str, str] = {}
    models: list[str] = []
    deployments: dict[str, str] = {}
    url_template: str | None = None  # 自定义 URL 模板，支持 {base_url} 和 {deployment} 占位符


class KeywordWeight(BaseModel):
    weight_a: float = 0.5
    weight_b: float = 0.5
    source: Literal["manual", "learned"] = "manual"


class MatchRule(BaseModel):
    type: Literal["pattern", "keyword", "length", "header", "model_hint"]
    field: str = "all_text"
    pattern: str | None = None
    extract: str | None = None
    keywords: dict[str, KeywordWeight] = {}
    threshold: float = 0.15
    min_chars: int = 0
    max_chars: int | None = None


class RoutingRule(BaseModel):
    name: str
    match: MatchRule
    model: str
    fallback_model: str | None = None
    tier: str | None = None


class PerformanceConfig(BaseModel):
    enabled: bool = False
    strategy: Literal["latency", "throughput"] = "latency"
    candidates: list[str] = []
    window_seconds: int = 3600
    min_samples: int = 5


class RoutingConfig(BaseModel):
    default_model: str
    global_override: str | None = None
    rules: list[RoutingRule] = []
    performance: PerformanceConfig = PerformanceConfig()


class CircuitBreakerConfig(BaseModel):
    failure_threshold: int = 3
    recovery_seconds: int = 300


class FallbackConfig(BaseModel):
    enabled: bool = True
    default_tier: str = "T1"
    tiers: dict[str, list[str]] = {}
    max_retries: int = 8
    circuit_breaker: CircuitBreakerConfig = CircuitBreakerConfig()
    retry_on_status: list[int] = [429, 500, 502, 503]


class MetadataConfig(BaseModel):
    enabled: bool = True
    format: Literal["footer", "header", "none"] = "footer"
    fields: list[str] = ["timestamp", "model", "elapsed", "tokens", "stop_reason"]


class StatsConfig(BaseModel):
    enabled: bool = True
    path: str = "./stats.jsonl"
    max_bytes: int = 10_485_760


class LearningConfig(BaseModel):
    enabled: bool = False
    alpha: float = 0.1
    min_weight: float = 0.1
    max_weight: float = 0.9
    max_keywords_per_update: int = 5
    protect_manual: bool = True


class Config(BaseModel):
    version: int = 1
    server: ServerConfig = ServerConfig()
    providers: dict[str, ProviderConfig] = {}
    routing: RoutingConfig = RoutingConfig(default_model="claude-sonnet-4-6")
    fallback: FallbackConfig = FallbackConfig()
    metadata: MetadataConfig = MetadataConfig()
    stats: StatsConfig = StatsConfig()
    learning: LearningConfig = LearningConfig()


# ── ENV var substitution ───────────────────────────────────────────

_ENV_PATTERN = re.compile(r"\$\{(\w+)\}")


def resolve_env_vars(raw: str) -> str:
    """Replace ${VAR_NAME} placeholders with environment variable values."""
    def _replace(m: re.Match[str]) -> str:
        name = m.group(1)
        value = os.environ.get(name, "")
        if not value:
            _logger.warning("environment variable %s is not set, using empty string", name)
        return value
    return _ENV_PATTERN.sub(_replace, raw)


# ── YAML Config Loader ────────────────────────────────────────────

def _parse_yaml_config(resolved_yaml: str) -> Config:
    """Parse resolved YAML string into a Config object."""
    from ruamel.yaml import YAML  # type: ignore[import-untyped]
    yml = YAML()
    data: dict[str, Any] | None = yml.load(resolved_yaml)  # type: ignore[assignment]
    if data is None:
        data = {}
    return Config(**data)


def load_config(path: str) -> Config:
    """Load YAML config, resolve ${ENV} vars, validate schema and model references."""
    raw = Path(path).read_text(encoding="utf-8")
    resolved = resolve_env_vars(raw)
    cfg = _parse_yaml_config(resolved)
    validate_model_references(cfg)
    return cfg


def validate_model_references(cfg: Config) -> None:
    """Cross-section validation: all model references must exist in some provider."""
    if not cfg.providers:
        return  # No providers defined, skip validation
    all_models = {m for p in cfg.providers.values() for m in p.models}

    if cfg.routing.default_model not in all_models:
        raise ValueError(
            f"routing.default_model references unknown model "
            f"'{cfg.routing.default_model}'"
        )

    if cfg.routing.global_override and cfg.routing.global_override not in all_models:
        raise ValueError(
            f"routing.global_override references unknown model "
            f"'{cfg.routing.global_override}'"
        )

    for rule in cfg.routing.rules:
        if rule.model != "__dynamic__" and rule.model not in all_models:
            raise ValueError(
                f"rule '{rule.name}' references unknown model '{rule.model}'"
            )
        if rule.fallback_model and rule.fallback_model != "__dynamic__" and rule.fallback_model not in all_models:
            raise ValueError(
                f"rule '{rule.name}' fallback_model references unknown model "
                f"'{rule.fallback_model}'"
            )

    for tier_name, models in cfg.fallback.tiers.items():
        for m in models:
            if m not in all_models:
                raise ValueError(
                    f"fallback tier '{tier_name}' references unknown model '{m}'"
                )


# ── Hot-reload config store ───────────────────────────────────────

_config_lock = threading.Lock()
_config: Config | None = None
_config_mtime: float = 0.0
_config_path: str = ""


def init_config(path: str) -> Config:
    """Initialize the config store. Call once at startup."""
    global _config_path, _config, _config_mtime
    _config_path = path
    cfg = _reload_config()
    if cfg is None:
        raise FileNotFoundError(f"Config file not found: {path}")
    return _config  # type: ignore[return-value]


def _reload_config() -> Config | None:
    """Reload config if file changed (mtime check)."""
    global _config, _config_mtime
    p = Path(_config_path)
    if not p.exists():
        return None
    mtime = p.stat().st_mtime
    if mtime == _config_mtime and _config is not None:
        return _config
    with open(p, "r", encoding="utf-8") as f:
        _lock_shared(f)
        try:
            raw = f.read()
        finally:
            _unlock(f)
    resolved = resolve_env_vars(raw)
    cfg = _parse_yaml_config(resolved)
    validate_model_references(cfg)
    _config = cfg
    _config_mtime = mtime
    return cfg


def get_config() -> Config:
    """Return current config, hot-reloading if file changed."""
    with _config_lock:
        if _config_path:
            _reload_config()
        if _config is None:
            return Config()
        return _config


def save_config(cfg: Config) -> None:
    """Atomically write config back to YAML (no comment preservation)."""
    global _config, _config_mtime
    p = Path(_config_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    from ruamel.yaml import YAML  # type: ignore[import-untyped]
    yml = YAML()
    yml.default_flow_style = False
    with open(tmp, "w", encoding="utf-8") as f:
        _lock_exclusive(f)
        try:
            yml.dump(cfg.model_dump(), f)
        finally:
            _unlock(f)
    os.replace(str(tmp), str(p))
    with _config_lock:
        _config = cfg
        _config_mtime = p.stat().st_mtime


def reset_config() -> None:
    """Reset config state. Intended for test teardown."""
    global _config, _config_path, _config_mtime
    _config = None
    _config_path = ""
    _config_mtime = 0.0


# ── Legacy backward compatibility ─────────────────────────────────
# These will be removed in Phase 2/3 when consumers migrate to get_config()

class Settings(BaseSettings):
    upstream_url: str = "http://localhost:8080"
    upstream_api_key: str = ""
    port: int = 8100
    rules_path: str = "./config.yaml"
    stats_path: str = "./stats.jsonl"
    openai_upstream_url: str = "http://localhost:8081"
    vertex_upstream_url: str = "http://localhost:8082"
    log_level: str = "warning"
    keyword_score_threshold: float = 0.15
    stats_max_bytes: int = 10_485_760
    fallback_failure_threshold: int = 3
    fallback_recovery_seconds: int = 300

    model_config = {"env_prefix": "LLM_ROUTER_"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
