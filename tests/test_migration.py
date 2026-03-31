"""Tests for JSON-to-YAML migration (migrations/json_to_yaml.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# migrations/ has no __init__.py — add to sys.path for direct import
_migrations_dir = str(Path(__file__).resolve().parent.parent / "migrations")
if _migrations_dir not in sys.path:
    sys.path.insert(0, _migrations_dir)

from json_to_yaml import (
    _build_fallback,
    _build_learning,
    _build_metadata,
    _build_providers,
    _build_routing_rules,
    migrate,
)


# ── Helper ─────────────────────────────────────────────────────────

def _representative_rules() -> dict:
    """A realistic rules.json payload for roundtrip tests."""
    return {
        "default_model": "claude-sonnet-4-6",
        "global_override": None,
        "force_overrides": [
            {"pattern": "FORCE_OPUS_NOW", "model": "claude-opus-4-6"},
        ],
        "cron_routes": {
            "a-stock-premarket": "claude-opus-4-6",
        },
        "keywords": {
            "深度分析": {
                "opus_weight": 0.8,
                "sonnet_weight": 0.2,
                "source": "manual",
                "tier": "T1",
            },
            "天气": {
                "opus_weight": 0.1,
                "sonnet_weight": 0.9,
                "source": "manual",
                "tier": "T2",
            },
        },
        "keyword_score_threshold": 0.15,
        "length_fallback": {
            "threshold": 2000,
            "short_model": "claude-sonnet-4-6",
            "long_model": "claude-opus-4-6",
        },
        "fallback": {
            "enabled": True,
            "default_tier": "T1",
            "tiers": {
                "T1": ["claude-opus-4-6", "claude-sonnet-4-6"],
                "T2": ["claude-sonnet-4-6"],
            },
            "max_retries": 8,
            "retry_on_status": [429, 500, 502, 503],
            "openai_models": ["gpt-4o"],
            "vertex_models": ["gemini-pro"],
            "deployments": {"gpt-4o": "gpt-4o-deploy"},
        },
        "inject_metadata": {
            "enabled": True,
            "include_timestamp": True,
            "include_model": True,
            "include_elapsed": True,
            "include_tokens": False,
            "include_reason": True,
        },
        "auto_learn": {
            "enabled": True,
            "alpha": 0.2,
            "min_weight": 0.05,
            "max_weight": 0.95,
            "max_keywords_per_update": 3,
            "protect_manual": False,
        },
    }


def _load_yaml(path: Path) -> dict:
    from ruamel.yaml import YAML
    yml = YAML()
    return yml.load(path.read_text(encoding="utf-8"))


# ── 1. Full migration roundtrip ───────────────────────────────────

class TestMigrateRoundtrip:
    def test_full_migration(self, tmp_path: Path):
        src = tmp_path / "rules.json"
        dst = tmp_path / "config.yaml"
        src.write_text(json.dumps(_representative_rules()), encoding="utf-8")

        migrate(str(src), str(dst))

        data = _load_yaml(dst)
        assert data["version"] == 1

        # providers
        assert "azure-openai" in data["providers"]
        assert "google-vertex" in data["providers"]
        assert "gpt-4o" in data["providers"]["azure-openai"]["models"]
        assert "gemini-pro" in data["providers"]["google-vertex"]["models"]

        # routing rules
        rules = data["routing"]["rules"]
        rule_names = [r["name"] for r in rules]
        assert "force-FORCE_OPUS_NOW" in rule_names
        assert "cron-a-stock-premarket" in rule_names

        # fallback
        assert "tiers" in data["fallback"]
        assert "circuit_breaker" in data["fallback"]
        # openai_models / vertex_models should NOT be in fallback
        assert "openai_models" not in data["fallback"]
        assert "vertex_models" not in data["fallback"]

        # metadata
        assert "timestamp" in data["metadata"]["fields"]
        assert "stop_reason" in data["metadata"]["fields"]

        # learning
        assert data["learning"]["enabled"] is True
        assert data["learning"]["alpha"] == 0.2

    def test_empty_rules_json(self, tmp_path: Path):
        src = tmp_path / "rules.json"
        dst = tmp_path / "config.yaml"
        src.write_text("{}", encoding="utf-8")

        migrate(str(src), str(dst))

        data = _load_yaml(dst)
        assert data["version"] == 1
        assert data["routing"]["default_model"] == "claude-sonnet-4-6"
        assert data["routing"]["rules"] == []

    def test_no_keywords_only_force_and_cron(self, tmp_path: Path):
        rules = {
            "default_model": "claude-sonnet-4-6",
            "force_overrides": [
                {"pattern": "FORCE", "model": "claude-opus-4-6"},
            ],
            "cron_routes": {"patrol": "claude-sonnet-4-6"},
        }
        src = tmp_path / "rules.json"
        dst = tmp_path / "config.yaml"
        src.write_text(json.dumps(rules), encoding="utf-8")

        migrate(str(src), str(dst))

        data = _load_yaml(dst)
        rule_names = [r["name"] for r in data["routing"]["rules"]]
        assert "force-FORCE" in rule_names
        assert "cron-patrol" in rule_names
        # No keyword rules
        assert all("keyword" not in n for n in rule_names)


# ── 2. Individual builder functions ────────────────────────────────

class TestBuildProviders:
    def test_with_openai_and_vertex(self):
        rules = {
            "default_model": "claude-sonnet-4-6",
            "fallback": {
                "openai_models": ["gpt-4o"],
                "vertex_models": ["gemini-pro"],
                "tiers": {"T1": ["claude-sonnet-4-6"]},
                "deployments": {"gpt-4o": "gpt4o-deploy"},
            },
        }
        providers = _build_providers(rules)
        assert "azure-openai" in providers
        assert "google-vertex" in providers
        assert "gpt-4o" in providers["azure-openai"]["models"]
        assert providers["azure-openai"]["deployments"]["gpt-4o"] == "gpt4o-deploy"
        assert "gemini-pro" in providers["google-vertex"]["models"]

    def test_anthropic_models_separate(self):
        rules = {
            "default_model": "claude-opus-4-6",
            "fallback": {
                "openai_models": ["gpt-4o"],
                "vertex_models": [],
                "tiers": {"T1": ["claude-opus-4-6", "gpt-4o"]},
            },
        }
        providers = _build_providers(rules)
        assert "anthropic-main" in providers
        assert "claude-opus-4-6" in providers["anthropic-main"]["models"]
        # claude-opus should NOT be in openai provider
        assert "claude-opus-4-6" not in providers["azure-openai"]["models"]


class TestBuildRoutingRules:
    def test_force_override_produces_pattern_rule(self):
        rules = {
            "force_overrides": [
                {"pattern": "MY_PATTERN", "model": "claude-opus-4-6"},
            ],
        }
        result = _build_routing_rules(rules)
        assert len(result) == 1
        assert result[0]["name"] == "force-MY_PATTERN"
        assert result[0]["match"]["type"] == "pattern"
        assert result[0]["model"] == "claude-opus-4-6"

    def test_keywords_produce_keyword_rules(self):
        rules = {
            "keywords": {
                "分析": {
                    "opus_weight": 0.8,
                    "sonnet_weight": 0.2,
                    "tier": "T1",
                },
            },
            "keyword_score_threshold": 0.2,
        }
        result = _build_routing_rules(rules)
        kw_rules = [r for r in result if r["match"]["type"] == "keyword"]
        assert len(kw_rules) == 1
        assert kw_rules[0]["name"] == "keyword-tier-t1"
        assert kw_rules[0]["match"]["threshold"] == 0.2

    def test_length_fallback_produces_two_rules(self):
        rules = {
            "length_fallback": {
                "threshold": 500,
                "short_model": "claude-sonnet-4-6",
                "long_model": "claude-opus-4-6",
            },
        }
        result = _build_routing_rules(rules)
        length_rules = [r for r in result if r["match"]["type"] == "length"]
        assert len(length_rules) == 2
        names = {r["name"] for r in length_rules}
        assert names == {"short-context", "long-context"}

        short = next(r for r in length_rules if r["name"] == "short-context")
        assert short["match"]["max_chars"] == 499
        assert short["model"] == "claude-sonnet-4-6"

        long = next(r for r in length_rules if r["name"] == "long-context")
        assert long["match"]["min_chars"] == 500
        assert long["model"] == "claude-opus-4-6"

    def test_cron_routes_produce_pattern_rules(self):
        rules = {
            "cron_routes": {"morning": "claude-opus-4-6"},
        }
        result = _build_routing_rules(rules)
        assert len(result) == 1
        assert result[0]["name"] == "cron-morning"
        assert result[0]["match"]["type"] == "pattern"
        assert result[0]["match"]["field"] == "system_prompt"


class TestBuildMetadata:
    def test_metadata_fields(self):
        rules = {
            "inject_metadata": {
                "enabled": True,
                "include_timestamp": True,
                "include_model": True,
                "include_elapsed": False,
                "include_tokens": True,
                "include_reason": False,
            },
        }
        meta = _build_metadata(rules)
        assert meta["enabled"] is True
        assert meta["format"] == "footer"
        assert meta["fields"] == ["timestamp", "model", "tokens"]

    def test_no_metadata_key(self):
        meta = _build_metadata({})
        assert meta["enabled"] is True
        assert meta["fields"] == []


class TestBuildLearning:
    def test_learning_params(self):
        rules = {
            "auto_learn": {
                "enabled": True,
                "alpha": 0.3,
                "min_weight": 0.05,
                "max_weight": 0.95,
                "max_keywords_per_update": 10,
                "protect_manual": False,
            },
        }
        learning = _build_learning(rules)
        assert learning["enabled"] is True
        assert learning["alpha"] == 0.3
        assert learning["protect_manual"] is False

    def test_no_auto_learn_key(self):
        learning = _build_learning({})
        assert learning["enabled"] is False
        assert learning["alpha"] == 0.1


class TestBuildFallback:
    def test_circuit_breaker_present(self):
        rules = {
            "fallback": {
                "enabled": True,
                "tiers": {"T1": ["m1"]},
                "max_retries": 5,
            },
        }
        fb = _build_fallback(rules)
        assert "circuit_breaker" in fb
        assert fb["circuit_breaker"]["failure_threshold"] == 3

    def test_openai_vertex_not_in_output(self):
        rules = {
            "fallback": {
                "openai_models": ["gpt-4o"],
                "vertex_models": ["gemini-pro"],
                "tiers": {"T1": ["m1"]},
            },
        }
        fb = _build_fallback(rules)
        assert "openai_models" not in fb
        assert "vertex_models" not in fb
        assert "deployments" not in fb

    def test_retry_on_status_preserved(self):
        rules = {
            "fallback": {
                "retry_on_status": [429, 503],
            },
        }
        fb = _build_fallback(rules)
        assert fb["retry_on_status"] == [429, 503]
