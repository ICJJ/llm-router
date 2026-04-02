"""/route command parsing and execution. Returns synthetic Anthropic response."""
from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from .config import get_config, save_config, RoutingRule, MatchRule, KeywordWeight
from .router import content_to_text as _content_to_text


MODEL_ALIASES = {
    # OpenAI — strongest first
    "gpt-5.4": "gpt-5.4",
    "o3": "o3",
    "gpt-4.1": "gpt-4.1",
    "o4-mini": "o4-mini",
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    # Gemini — strongest first
    "gemini-3.1-pro": "gemini-3.1-pro-preview",
    "gemini-3.1-pro-preview": "gemini-3.1-pro-preview",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-flash": "gemini-2.5-flash",
    "gemini-3.1-flash-lite": "gemini-3.1-flash-lite-preview",
    "gemini-3.1-flash-lite-preview": "gemini-3.1-flash-lite-preview",
    # Shortcuts
    "best": "gpt-5.4",
    "reason": "o3",
    "fast": "gemini-2.5-flash",
    "stable": "gpt-4.1",
}


def is_route_command(messages: list[dict[str, Any]]) -> bool:
    """Check if last user message starts with /route."""
    for m in reversed(messages):
        if m.get("role") == "user":
            text = _content_to_text(m.get("content", ""))
            return text.strip().startswith("/route")
    return False


def execute(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Execute /route command and return synthetic Anthropic API response."""
    text = _extract_command_text(messages)
    parts = text.split()
    if len(parts) < 2:
        return _synth("用法: /route <add|del|list|force|stats|reset-learn|config|override|health>")
    return _dispatch_command(parts[1], parts, text)


def _extract_command_text(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return _content_to_text(m.get("content", "")).strip()
    return ""


def _parse_stats_count(parts: list[str]) -> int:
    if len(parts) >= 3 and parts[2].isdigit():
        return int(parts[2])
    return 10


_COMMAND_TABLE: dict[str, tuple[int, Callable[[list[str]], dict[str, Any]]]] = {
    "list": (0, lambda parts: _cmd_list()),
    "add": (4, lambda parts: _cmd_add(parts[2], parts[3])),
    "del": (3, lambda parts: _cmd_del(parts[2])),
    "force": (0, lambda parts: _cmd_force(parts[2:])),
    "stats": (0, lambda parts: _cmd_stats(_parse_stats_count(parts))),
    "reset-learn": (0, lambda parts: _cmd_reset_learn()),
    "config": (4, lambda parts: _cmd_config(parts[2], parts[3])),
    "override": (2, lambda parts: _handle_override(parts)),
    "health": (1, lambda parts: _handle_health(parts)),
}


def _dispatch_command(sub: str, parts: list[str], text: str) -> dict[str, Any]:
    entry = _COMMAND_TABLE.get(sub)
    if entry is None:
        return _synth(f"未知命令: {text}\n用法: /route <add|del|list|force|stats|reset-learn|config|override|health>")
    min_args, handler = entry
    if min_args and len(parts) < min_args:
        return _synth(f"未知命令: {text}\n用法: /route <add|del|list|force|stats|reset-learn|config|override|health>")
    return handler(parts)


def _cmd_list() -> dict[str, Any]:
    cfg = get_config()
    lines: list[str] = [
        "📋 **LLM Router 规则**\n",
        f"**默认模型:** {cfg.routing.default_model}",
        "",
    ]
    # Force/pattern rules
    pattern_rules = [r for r in cfg.routing.rules if r.match.type == "pattern"]
    if pattern_rules:
        lines.append("**路由规则:**")
        for r in pattern_rules:
            lines.append(f"  • `{r.match.pattern}` → {r.model}")
        lines.append("")
    # Keyword rules
    for r in cfg.routing.rules:
        if r.match.type == "keyword" and r.match.keywords:
            lines.append(f"**关键词:** {len(r.match.keywords)} 条")
            for kw, w in r.match.keywords.items():
                dominant = "a" if w.weight_a > w.weight_b else "b"
                lines.append(f"  • {kw} → {dominant} (a={w.weight_a:.1f}/b={w.weight_b:.1f}) [{w.source}]")
            lines.append("")
    # Metadata
    lines.append(f"**元信息注入:** {'开启' if cfg.metadata.enabled else '关闭'}")
    return _synth("\n".join(lines))


def _cmd_add(keyword: str, model_alias: str) -> dict[str, Any]:
    model = MODEL_ALIASES.get(model_alias, model_alias if model_alias in MODEL_ALIASES.values() else None)
    if not model:
        shortcuts = "best, reason, fast, stable, gpt-5.4, o3, gemini-2.5-flash, gpt-4.1"
        return _synth(f"❌ 未知模型: {model_alias}\n可选: {shortcuts}")
    cfg = get_config()
    # Find first keyword-type rule
    kw_rule = None
    for rule in cfg.routing.rules:
        if rule.match.type == "keyword":
            kw_rule = rule
            break
    if kw_rule is None:
        kw_rule = RoutingRule(
            name="keywords",
            match=MatchRule(type="keyword", field="all_text"),
            model=cfg.routing.default_model,
            fallback_model="gpt-4o",
        )
        cfg.routing.rules.append(kw_rule)
    # Higher weight_a = prefer model A (the target model)
    kw_rule.match.keywords[keyword] = KeywordWeight(
        weight_a=0.8,
        weight_b=0.2,
        source="manual",
    )
    save_config(cfg)
    side_model = kw_rule.model
    return _synth(f"✅ 已添加关键词: {keyword} (偏向 {side_model})")


def _cmd_del(keyword: str) -> dict[str, Any]:
    cfg = get_config()
    for rule in cfg.routing.rules:
        if rule.match.type == "keyword" and keyword in rule.match.keywords:
            del rule.match.keywords[keyword]
            save_config(cfg)
            return _synth(f"✅ 已删除关键词: {keyword}")
    return _synth(f"❌ 关键词不存在: {keyword}")


def _cmd_force(args: list[str]) -> dict[str, Any]:
    if len(args) >= 2 and args[0] == "del":
        return _cmd_force_del(args[1])
    if len(args) >= 2:
        return _cmd_force_set(args[0], args[1])
    return _synth("用法: /route force <pattern> <opus|sonnet>\n      /route force del <pattern>")


def _cmd_force_set(pattern: str, model_alias: str) -> dict[str, Any]:
    if len(pattern) < 2:
        return _synth("❌ pattern 长度至少 2 个字符")
    model = MODEL_ALIASES.get(model_alias, model_alias if model_alias in MODEL_ALIASES.values() else None)
    if not model:
        shortcuts = "best, reason, fast, stable, gpt-5.4, o3, gemini-2.5-flash, gpt-4.1"
        return _synth(f"❌ 未知模型: {model_alias}\n可选: {shortcuts}")
    cfg = get_config()
    # Update existing or add new pattern rule
    for rule in cfg.routing.rules:
        if rule.match.type == "pattern" and rule.match.pattern == pattern:
            rule.model = model
            save_config(cfg)
            return _synth(f"✅ 已更新强制覆盖: {pattern} → {model}")
    new_rule = RoutingRule(
        name=f"force-{pattern.replace(':', '-')}",
        match=MatchRule(type="pattern", field="system_prompt", pattern=pattern),
        model=model,
    )
    cfg.routing.rules.insert(0, new_rule)  # Highest priority
    save_config(cfg)
    return _synth(f"✅ 已添加强制覆盖: {pattern} → {model}")


def _cmd_force_del(pattern: str) -> dict[str, Any]:
    cfg = get_config()
    before = len(cfg.routing.rules)
    cfg.routing.rules = [
        r for r in cfg.routing.rules
        if not (r.match.type == "pattern" and r.match.pattern == pattern)
    ]
    if len(cfg.routing.rules) < before:
        save_config(cfg)
        return _synth(f"✅ 已删除强制覆盖: {pattern}")
    return _synth(f"❌ 未找到强制覆盖: {pattern}")


def _cmd_stats(count: int) -> dict[str, Any]:
    from . import stats as stats_mod
    entries = stats_mod.read_recent(count)
    if not entries:
        return _synth("📊 暂无路由统计数据")

    lines = [f"📊 **最近 {len(entries)} 条路由记录**\n"]
    for e in entries:
        ts = e.get("ts", "?")[:19]
        routed = e.get("model_routed", "?")
        reason = e.get("route_reason", "?")
        elapsed = e.get("elapsed_ms", 0)
        out_tok = e.get("output_tokens", 0)
        short_model = "O" if "opus" in routed else "S"
        lines.append(f"  {ts} | {short_model} | {elapsed}ms | {out_tok}tok | {reason}")

    return _synth("\n".join(lines))


def _cmd_reset_learn() -> dict[str, Any]:
    cfg = get_config()
    removed = 0
    for rule in cfg.routing.rules:
        if rule.match.type == "keyword":
            for kw in list(rule.match.keywords.keys()):
                if rule.match.keywords[kw].source == "learned":
                    del rule.match.keywords[kw]
                    removed += 1
    save_config(cfg)
    return _synth(f"✅ 已清除 {removed} 条自动学习规则")


def _cmd_config(key: str, value: str) -> dict[str, Any]:
    if key == "inject":
        cfg = get_config()
        cfg.metadata.enabled = value.lower() in ("on", "true", "1")
        save_config(cfg)
        status = "开启" if cfg.metadata.enabled else "关闭"
        return _synth(f"✅ 元信息注入已{status}")
    return _synth(f"❌ 未知配置项: {key}\n可选: inject")


# ── helpers ─────────────────────────────────────────────────────


def _synth(text: str) -> dict[str, Any]:
    """Build a synthetic Anthropic Messages API response."""
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": "llm-router",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def _handle_override(parts: list[str]) -> dict[str, Any]:
    if len(parts) < 3:
        return _synth("用法: /route override <model|off>")
    target = parts[2]
    cfg = get_config()
    if target == "off":
        cfg.routing.global_override = None
        save_config(cfg)
        return _synth("✅ 全局覆盖已关闭，恢复正常路由")
    model = MODEL_ALIASES.get(target, target)
    cfg.routing.global_override = model
    save_config(cfg)
    return _synth(f"✅ 全局覆盖已设置: 所有请求将使用 {model}")


def _handle_health(parts: list[str]) -> dict[str, Any]:
    from . import fallback
    status = fallback.get_health_status()
    lines = ["\U0001F4CA 模型健康状态:\n"]
    for model, info in status.items():
        icon = "\U0001F7E2" if info["healthy"] else "\U0001F534"
        lines.append(f"{icon} {model}: failures={info['consecutive_failures']}")
    if len(lines) == 1:
        lines.append("暂无模型健康数据")
    return _synth("\n".join(lines))
