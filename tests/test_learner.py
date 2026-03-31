"""Tests for auto-learner."""
import asyncio
from pathlib import Path
from typing import Any

import pytest
from ruamel.yaml import YAML  # type: ignore[import-untyped]

from llm_router.config import init_config, get_config, save_config, KeywordWeight
import llm_router.config as config_mod
from llm_router.learner import learn


@pytest.fixture(autouse=True)
def _init_config(tmp_path: Path) -> None:  # pyright: ignore[reportUnusedFunction]
    cfg_data = {
        "routing": {
            "default_model": "claude-sonnet-4-6",
            "rules": [
                {
                    "name": "keywords",
                    "match": {
                        "type": "keyword",
                        "field": "all_text",
                        "keywords": {
                            "manual_kw": {
                                "weight_a": 0.5,
                                "weight_b": 0.5,
                                "source": "manual",
                            },
                            "auto_kw": {
                                "weight_a": 0.5,
                                "weight_b": 0.5,
                                "source": "learned",
                            },
                        },
                    },
                    "model": "claude-opus-4-6",
                    "fallback_model": "claude-sonnet-4-6",
                },
            ],
        },
        "learning": {
            "enabled": True,
            "alpha": 0.1,
            "min_weight": 0.1,
            "max_weight": 0.9,
            "max_keywords_per_update": 5,
            "protect_manual": True,
        },
    }
    cfg_path = tmp_path / "config.yaml"
    yml = YAML()
    yml.dump(cfg_data, cfg_path)
    init_config(str(cfg_path))
    yield
    config_mod._config = None
    config_mod._config_mtime = 0.0
    config_mod._config_path = ""


def _get_keyword(name: str) -> KeywordWeight:
    """Helper to find a keyword weight across all keyword rules."""
    cfg = get_config()
    for rule in cfg.routing.rules:
        if rule.match.type == "keyword" and name in rule.match.keywords:
            return rule.match.keywords[name]
    raise KeyError(f"keyword '{name}' not found")


def test_learn_protects_manual():
    """Manual keywords should not be updated."""
    messages = [{"role": "user", "content": "manual_kw auto_kw 分析"}]
    response: dict[str, Any] = {
        "usage": {"output_tokens": 5000},
        "content": [{"type": "text", "text": "long response"}],
    }
    asyncio.run(learn(messages, response, elapsed_s=5.0))
    # manual should be untouched
    manual = _get_keyword("manual_kw")
    assert manual.source == "manual"
    assert manual.weight_a == 0.5
    # auto should be updated towards opus (high output tokens)
    auto = _get_keyword("auto_kw")
    assert auto.source == "learned"
    assert auto.weight_a > 0.5


def test_learn_favours_sonnet_for_simple():
    """Short responses should nudge weights toward sonnet."""
    messages = [{"role": "user", "content": "auto_kw 查一下"}]
    response: dict[str, Any] = {
        "usage": {"output_tokens": 200},
        "content": [{"type": "text", "text": "short"}],
    }
    asyncio.run(learn(messages, response, elapsed_s=1.0))
    auto = _get_keyword("auto_kw")
    assert auto.weight_b > 0.5


def test_learn_disabled():
    """When disabled, no weights should change."""
    cfg = get_config()
    cfg.learning.enabled = False
    save_config(cfg)

    messages = [{"role": "user", "content": "auto_kw test"}]
    response: dict[str, Any] = {
        "usage": {"output_tokens": 5000},
        "content": [],
    }
    asyncio.run(learn(messages, response, elapsed_s=5.0))
    auto = _get_keyword("auto_kw")
    assert auto.weight_a == 0.5


def test_learn_weight_bounds():
    """Weights should stay within [min_weight, max_weight]."""
    # Push weight to extreme
    cfg = get_config()
    for rule in cfg.routing.rules:
        if rule.match.type == "keyword":
            rule.match.keywords["auto_kw"].weight_a = 0.89
    save_config(cfg)

    messages = [{"role": "user", "content": "auto_kw"}]
    response: dict[str, Any] = {
        "usage": {"output_tokens": 5000},
        "content": [{"type": "text", "text": "x" * 5000}],
    }
    asyncio.run(learn(messages, response, elapsed_s=5.0))
    auto = _get_keyword("auto_kw")
    assert auto.weight_a <= 0.9


def test_learn_concurrent_no_data_loss():
    """Concurrent learn() calls must not overwrite each other's updates."""
    cfg = get_config()
    for rule in cfg.routing.rules:
        if rule.match.type == "keyword":
            for i in range(5):
                rule.match.keywords[f"conc_kw_{i}"] = KeywordWeight(
                    weight_a=0.5, weight_b=0.5, source="learned",
                )
    save_config(cfg)

    async def _run() -> None:
        tasks: list[asyncio.Task[None]] = []
        for i in range(5):
            msgs: list[dict[str, Any]] = [{"role": "user", "content": f"conc_kw_{i}"}]
            resp: dict[str, Any] = {
                "usage": {"output_tokens": 5000},
                "content": [{"type": "text", "text": "big output"}],
            }
            tasks.append(asyncio.ensure_future(learn(msgs, resp, elapsed_s=5.0)))
        await asyncio.gather(*tasks)

    asyncio.run(_run())
    for i in range(5):
        kw = _get_keyword(f"conc_kw_{i}")
        assert kw.weight_a > 0.5, f"conc_kw_{i} was not updated — concurrency bug"
