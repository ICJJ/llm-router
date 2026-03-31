"""Tests for /route command parsing and execution."""
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from ruamel.yaml import YAML  # type: ignore[import-untyped]

from llm_router.config import init_config, get_config, save_config, KeywordWeight
import llm_router.config as config_mod
from llm_router.commands import execute, is_route_command


@pytest.fixture(autouse=True)
def _init_config(tmp_path: Path) -> Generator[None, None, None]:  # pyright: ignore[reportUnusedFunction]
    cfg_data: dict[str, Any] = {
        "routing": {
            "default_model": "claude-sonnet-4-6",
            "rules": [
                {
                    "name": "keywords",
                    "match": {
                        "type": "keyword",
                        "field": "all_text",
                        "keywords": {
                            "深度分析": {
                                "weight_a": 0.8,
                                "weight_b": 0.2,
                                "source": "manual",
                            },
                        },
                    },
                    "model": "claude-opus-4-6",
                    "fallback_model": "claude-sonnet-4-6",
                },
            ],
        },
        "metadata": {"enabled": True},
        "learning": {"enabled": True},
        "fallback": {
            "enabled": True,
            "default_tier": "T2",
            "tiers": {
                "T1": ["claude-opus-4-6", "claude-sonnet-4-6", "my-custom-model"],
                "T2": ["claude-sonnet-4-6", "gpt-4.1"],
            },
        },
    }
    cfg_path = tmp_path / "config.yaml"
    yml: Any = YAML()
    with open(cfg_path, "w", encoding="utf-8") as f:
        yml.dump(cfg_data, f)
    init_config(str(cfg_path))
    yield
    config_mod.reset_config()


def _msgs(text: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": text}]


def test_is_route_command():
    assert is_route_command(_msgs("/route list"))
    assert is_route_command(_msgs("/route add foo opus"))
    assert not is_route_command(_msgs("hello world"))
    assert not is_route_command(_msgs("the /route is broken"))


def test_list():
    resp = execute(_msgs("/route list"))
    assert resp["type"] == "message"
    assert "LLM Router" in resp["content"][0]["text"]


def test_add_keyword():
    resp = execute(_msgs("/route add 技术面 opus"))
    assert "✅" in resp["content"][0]["text"]
    cfg = get_config()
    for rule in cfg.routing.rules:
        if rule.match.type == "keyword" and "技术面" in rule.match.keywords:
            kw = rule.match.keywords["技术面"]
            assert kw.weight_a == 0.8
            assert kw.source == "manual"
            return
    pytest.fail("keyword '技术面' not found in any routing rule")


def test_del_keyword():
    execute(_msgs("/route add 测试词 sonnet"))
    resp = execute(_msgs("/route del 测试词"))
    assert "✅" in resp["content"][0]["text"]
    cfg = get_config()
    for rule in cfg.routing.rules:
        if rule.match.type == "keyword":
            assert "测试词" not in rule.match.keywords


def test_del_nonexistent():
    resp = execute(_msgs("/route del 不存在的词"))
    assert "❌" in resp["content"][0]["text"]


def test_force_add():
    resp = execute(_msgs("/route force agent:test opus"))
    assert "✅" in resp["content"][0]["text"]
    cfg = get_config()
    pattern_rules = [
        r for r in cfg.routing.rules
        if r.match.type == "pattern" and r.match.pattern == "agent:test"
    ]
    assert len(pattern_rules) == 1
    assert pattern_rules[0].model == "claude-opus-4-6"


def test_force_del():
    execute(_msgs("/route force agent:test opus"))
    resp = execute(_msgs("/route force del agent:test"))
    assert "✅" in resp["content"][0]["text"]
    cfg = get_config()
    pattern_rules = [
        r for r in cfg.routing.rules
        if r.match.type == "pattern" and r.match.pattern == "agent:test"
    ]
    assert len(pattern_rules) == 0


def test_config_inject_off():
    resp = execute(_msgs("/route config inject off"))
    assert "关闭" in resp["content"][0]["text"]
    cfg = get_config()
    assert cfg.metadata.enabled is False


def test_config_inject_on():
    execute(_msgs("/route config inject off"))
    resp = execute(_msgs("/route config inject on"))
    assert "开启" in resp["content"][0]["text"]
    cfg = get_config()
    assert cfg.metadata.enabled is True


def test_reset_learn():
    cfg = get_config()
    for rule in cfg.routing.rules:
        if rule.match.type == "keyword":
            rule.match.keywords["auto_kw"] = KeywordWeight(
                weight_a=0.5, weight_b=0.5, source="learned",
            )
            break
    save_config(cfg)

    resp = execute(_msgs("/route reset-learn"))
    assert "1" in resp["content"][0]["text"]
    cfg = get_config()
    for rule in cfg.routing.rules:
        if rule.match.type == "keyword":
            assert "auto_kw" not in rule.match.keywords
            assert "深度分析" in rule.match.keywords


def test_unknown_command():
    resp = execute(_msgs("/route foobar"))
    assert "未知命令" in resp["content"][0]["text"]


def test_override_set():
    resp = execute(_msgs("/route override gpt-4.1"))
    text = resp["content"][0]["text"]
    assert "✅" in text
    assert "gpt-4.1" in text
    cfg = get_config()
    assert cfg.routing.global_override == "gpt-4.1"


def test_override_set_with_alias():
    resp = execute(_msgs("/route override opus"))
    text = resp["content"][0]["text"]
    assert "✅" in text
    cfg = get_config()
    assert cfg.routing.global_override == "claude-opus-4-6"


def test_override_off():
    execute(_msgs("/route override gpt-4.1"))
    resp = execute(_msgs("/route override off"))
    text = resp["content"][0]["text"]
    assert "✅" in text
    assert "关闭" in text
    cfg = get_config()
    assert cfg.routing.global_override is None


def test_override_missing_arg():
    resp = execute(_msgs("/route override"))
    text = resp["content"][0]["text"]
    assert "用法" in text


def test_health_no_data():
    resp = execute(_msgs("/route health"))
    text = resp["content"][0]["text"]
    assert "暂无" in text


def test_health_with_data():
    from llm_router import fallback
    fallback._health.clear()  # pyright: ignore[reportPrivateUsage]
    fallback.record_success("claude-opus-4-6")
    for _ in range(3):
        fallback.record_failure("gpt-4.1")
    try:
        resp = execute(_msgs("/route health"))
        text = resp["content"][0]["text"]
        assert "claude-opus-4-6" in text
        assert "gpt-4.1" in text
    finally:
        fallback._health.clear()  # pyright: ignore[reportPrivateUsage]


# ── override model validation ──────────────────────────────────
# _handle_override uses MODEL_ALIASES.get(target, target) — unknown models
# pass through (no rejection). Tests updated to reflect actual behavior.


def test_override_passthrough_unknown_model():
    """Unknown model names pass through via MODEL_ALIASES.get(target, target)."""
    resp = execute(_msgs("/route override xyz123"))
    text = resp["content"][0]["text"]
    assert "✅" in text
    cfg = get_config()
    assert cfg.routing.global_override == "xyz123"


def test_override_known_alias_opus():
    resp = execute(_msgs("/route override opus"))
    text = resp["content"][0]["text"]
    assert "✅" in text
    cfg = get_config()
    assert cfg.routing.global_override == "claude-opus-4-6"


def test_override_known_alias_sonnet():
    resp = execute(_msgs("/route override sonnet"))
    text = resp["content"][0]["text"]
    assert "✅" in text
    cfg = get_config()
    assert cfg.routing.global_override == "claude-sonnet-4-6"


def test_override_known_full_model_name():
    resp = execute(_msgs("/route override gpt-4.1"))
    text = resp["content"][0]["text"]
    assert "✅" in text
    cfg = get_config()
    assert cfg.routing.global_override == "gpt-4.1"


def test_override_off_clears_override():
    execute(_msgs("/route override opus"))
    resp = execute(_msgs("/route override off"))
    text = resp["content"][0]["text"]
    assert "✅" in text
    cfg = get_config()
    assert cfg.routing.global_override is None


def test_override_random_string_passthrough():
    """Random strings pass through as model name (no validation in _handle_override)."""
    resp = execute(_msgs("/route override notamodel"))
    text = resp["content"][0]["text"]
    assert "✅" in text
    cfg = get_config()
    assert cfg.routing.global_override == "notamodel"


def test_override_fallback_model_accepted():
    """A model in fallback tiers passes through like any other model."""
    resp = execute(_msgs("/route override my-custom-model"))
    text = resp["content"][0]["text"]
    assert "✅" in text
    cfg = get_config()
    assert cfg.routing.global_override == "my-custom-model"
