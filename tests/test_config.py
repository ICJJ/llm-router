"""Tests for YAML-based configuration (llm_router.config)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

import llm_router.config as cfg_mod
from llm_router.config import (
    Config,
    FallbackConfig,
    MatchRule,
    ProviderConfig,
    RoutingConfig,
    RoutingRule,
    _resolve_env_vars,
    _validate_model_references,
    get_config,
    init_config,
    load_config,
    save_config,
)


# ── Fixture: reset module-level globals between tests ──────────────

@pytest.fixture(autouse=True)
def _reset_config_globals():
    """Reset module globals so each test starts clean."""
    yield
    cfg_mod._config = None
    cfg_mod._config_path = ""
    cfg_mod._config_mtime = 0.0


# ── 1. Pydantic model defaults & validation ────────────────────────

class TestPydanticModels:
    def test_config_defaults(self):
        cfg = Config()
        assert cfg.version == 1
        assert cfg.server.port == 8100
        assert cfg.routing.default_model == "claude-sonnet-4-6"
        assert cfg.server.host == "0.0.0.0"
        assert cfg.server.log_level == "warning"

    def test_provider_config_requires_type_and_base_url(self):
        with pytest.raises(ValidationError):
            ProviderConfig()  # type: ignore[call-arg]

    def test_provider_config_valid(self):
        p = ProviderConfig(type="anthropic", base_url="https://example.com")
        assert p.type == "anthropic"
        assert p.base_url == "https://example.com"
        assert p.models == []

    def test_routing_rule_requires_name_match_model(self):
        with pytest.raises(ValidationError):
            RoutingRule()  # type: ignore[call-arg]

    def test_routing_rule_valid(self):
        r = RoutingRule(
            name="test",
            match=MatchRule(type="keyword"),
            model="claude-sonnet-4-6",
        )
        assert r.name == "test"
        assert r.model == "claude-sonnet-4-6"

    def test_invalid_log_level_raises(self):
        with pytest.raises(ValidationError):
            Config(server={"log_level": "invalid"})

    def test_invalid_auth_type_raises(self):
        with pytest.raises(ValidationError):
            ProviderConfig(
                type="anthropic",
                base_url="https://x.com",
                auth={"type": "invalid"},
            )


# ── 2. _resolve_env_vars ───────────────────────────────────────────

class TestResolveEnvVars:
    def test_existing_var_replaced(self, monkeypatch):
        monkeypatch.setenv("MY_TEST_VAR", "hello")
        assert _resolve_env_vars("${MY_TEST_VAR}") == "hello"

    def test_nonexistent_var_replaced_with_empty(self):
        key = "ABSOLUTELY_NONEXISTENT_VAR_12345"
        os.environ.pop(key, None)
        assert _resolve_env_vars(f"${{{key}}}") == ""

    def test_no_pattern_unchanged(self):
        assert _resolve_env_vars("plain text") == "plain text"

    def test_multiple_vars(self, monkeypatch):
        monkeypatch.setenv("A_VAR", "foo")
        monkeypatch.setenv("B_VAR", "bar")
        result = _resolve_env_vars("${A_VAR}:${B_VAR}")
        assert result == "foo:bar"

    def test_mixed_existing_and_missing(self, monkeypatch):
        monkeypatch.setenv("EXIST_VAR", "yes")
        os.environ.pop("MISS_VAR", None)
        result = _resolve_env_vars("${EXIST_VAR}-${MISS_VAR}")
        assert result == "yes-"


# ── 3. load_config ─────────────────────────────────────────────────

class TestLoadConfig:
    def test_load_valid_yaml(self, tmp_path: Path):
        yaml_content = """\
version: 1
server:
  port: 9090
  log_level: debug
routing:
  default_model: my-model
"""
        p = tmp_path / "config.yaml"
        p.write_text(yaml_content, encoding="utf-8")
        cfg = load_config(str(p))
        assert cfg.version == 1
        assert cfg.server.port == 9090
        assert cfg.server.log_level == "debug"
        assert cfg.routing.default_model == "my-model"

    def test_load_with_env_var_substitution(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("TEST_PORT", "7777")
        yaml_content = """\
version: 1
server:
  port: ${TEST_PORT}
routing:
  default_model: test-model
"""
        p = tmp_path / "config.yaml"
        p.write_text(yaml_content, encoding="utf-8")
        cfg = load_config(str(p))
        assert cfg.server.port == 7777

    def test_empty_yaml_returns_defaults(self, tmp_path: Path):
        p = tmp_path / "config.yaml"
        p.write_text("", encoding="utf-8")
        cfg = load_config(str(p))
        assert cfg.version == 1
        assert cfg.server.port == 8100

    def test_invalid_yaml_structure_raises(self, tmp_path: Path):
        yaml_content = """\
version: 1
server:
  log_level: banana
routing:
  default_model: x
"""
        p = tmp_path / "config.yaml"
        p.write_text(yaml_content, encoding="utf-8")
        with pytest.raises(ValidationError):
            load_config(str(p))


# ── 4. _validate_model_references ──────────────────────────────────

class TestValidateModelReferences:
    def _make_config(self, **overrides) -> Config:
        """Helper to build Config with providers that list known models."""
        defaults = dict(
            providers={
                "main": ProviderConfig(
                    type="anthropic",
                    base_url="https://x.com",
                    models=["model-a", "model-b"],
                )
            },
            routing=RoutingConfig(default_model="model-a"),
        )
        defaults.update(overrides)
        return Config(**defaults)

    def test_valid_refs_no_error(self):
        cfg = self._make_config()
        _validate_model_references(cfg)  # should not raise

    def test_no_providers_skips_validation(self):
        cfg = Config()  # no providers
        _validate_model_references(cfg)  # should not raise

    def test_default_model_unknown_raises(self):
        cfg = self._make_config(
            routing=RoutingConfig(default_model="nonexistent"),
        )
        with pytest.raises(ValueError, match="routing.default_model"):
            _validate_model_references(cfg)

    def test_routing_rule_unknown_model_raises(self):
        cfg = self._make_config(
            routing=RoutingConfig(
                default_model="model-a",
                rules=[
                    RoutingRule(
                        name="bad",
                        match=MatchRule(type="keyword"),
                        model="unknown-model",
                    )
                ],
            ),
        )
        with pytest.raises(ValueError, match="rule 'bad' references unknown model"):
            _validate_model_references(cfg)

    def test_fallback_tier_unknown_model_raises(self):
        cfg = self._make_config(
            fallback=FallbackConfig(tiers={"T1": ["model-a", "ghost"]}),
        )
        with pytest.raises(ValueError, match="fallback tier 'T1' references unknown model"):
            _validate_model_references(cfg)

    def test_global_override_unknown_raises(self):
        cfg = self._make_config(
            routing=RoutingConfig(
                default_model="model-a",
                global_override="phantom",
            ),
        )
        with pytest.raises(ValueError, match="routing.global_override"):
            _validate_model_references(cfg)

    def test_fallback_model_unknown_raises(self):
        cfg = self._make_config(
            routing=RoutingConfig(
                default_model="model-a",
                rules=[
                    RoutingRule(
                        name="fb-bad",
                        match=MatchRule(type="keyword"),
                        model="model-a",
                        fallback_model="nonexist",
                    )
                ],
            ),
        )
        with pytest.raises(ValueError, match="fallback_model references unknown model"):
            _validate_model_references(cfg)


# ── 5. init_config / get_config / hot-reload ───────────────────────

class TestInitGetHotReload:
    @staticmethod
    def _write_yaml(path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8")

    def test_init_config_and_get_config(self, tmp_path: Path):
        p = tmp_path / "config.yaml"
        self._write_yaml(p, "version: 1\nrouting:\n  default_model: m1\n")
        cfg = init_config(str(p))
        assert cfg.routing.default_model == "m1"
        assert get_config().routing.default_model == "m1"

    def test_init_config_nonexistent_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            init_config(str(tmp_path / "nope.yaml"))

    def test_get_config_before_init_returns_defaults(self):
        cfg = get_config()
        assert cfg.version == 1
        assert cfg.server.port == 8100

    def test_hot_reload_picks_up_changes(self, tmp_path: Path):
        p = tmp_path / "config.yaml"
        self._write_yaml(p, "version: 1\nrouting:\n  default_model: old\n")
        init_config(str(p))
        assert get_config().routing.default_model == "old"

        # Modify file and bump mtime (Windows timer resolution workaround)
        import time
        time.sleep(0.05)
        self._write_yaml(p, "version: 1\nrouting:\n  default_model: new\n")
        os.utime(str(p), (time.time() + 1, time.time() + 1))

        assert get_config().routing.default_model == "new"


# ── 6. save_config ─────────────────────────────────────────────────

class TestSaveConfig:
    def test_save_and_reload(self, tmp_path: Path):
        p = tmp_path / "config.yaml"
        p.write_text("version: 1\nrouting:\n  default_model: original\n", encoding="utf-8")
        init_config(str(p))

        new_cfg = Config(routing=RoutingConfig(default_model="saved-model"))
        save_config(new_cfg)

        # Verify file content
        from ruamel.yaml import YAML
        yml = YAML()
        data = yml.load(p.read_text(encoding="utf-8"))
        assert data["routing"]["default_model"] == "saved-model"

    def test_get_config_after_save_returns_saved(self, tmp_path: Path):
        p = tmp_path / "config.yaml"
        p.write_text("version: 1\nrouting:\n  default_model: before\n", encoding="utf-8")
        init_config(str(p))

        new_cfg = Config(routing=RoutingConfig(default_model="after"))
        save_config(new_cfg)
        assert get_config().routing.default_model == "after"
