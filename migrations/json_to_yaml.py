"""Migrate rules.json + Settings env vars → config.yaml (YAML config format)."""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any


def build_providers(rules: dict[str, Any]) -> dict[str, Any]:
    """② Generate providers block from fallback lists + upstream URLs."""
    providers: OrderedDict[str, Any] = OrderedDict()
    fallback = rules.get("fallback", {})

    # Anthropic provider — all models not in openai/vertex lists
    openai_models = set(fallback.get("openai_models", []))
    vertex_models = set(fallback.get("vertex_models", []))

    # Collect all models from tiers
    all_tier_models: set[str] = set()
    for models in fallback.get("tiers", {}).values():
        all_tier_models.update(models)

    # Also include models from force_overrides and cron_routes
    for fo in rules.get("force_overrides", []):
        all_tier_models.add(fo["model"])
    for model in rules.get("cron_routes", {}).values():
        all_tier_models.add(model)

    # length_fallback models
    lf = rules.get("length_fallback", {})
    if lf.get("short_model"):
        all_tier_models.add(lf["short_model"])
    if lf.get("long_model"):
        all_tier_models.add(lf["long_model"])

    # default_model
    if rules.get("default_model"):
        all_tier_models.add(rules["default_model"])

    anthropic_models = sorted(
        all_tier_models - openai_models - vertex_models
    )

    if anthropic_models:
        providers["anthropic-main"] = OrderedDict([
            ("type", "anthropic"),
            ("base_url", "${LLM_ROUTER_UPSTREAM_URL}"),
            ("auth", OrderedDict([
                ("type", "header_key"),
                ("header", "x-api-key"),
                ("value", "${LLM_ROUTER_UPSTREAM_API_KEY}"),
            ])),
            ("models", anthropic_models),
        ])

    if openai_models:
        deployments = fallback.get("deployments", {})
        openai_list = sorted(openai_models)
        provider_data: dict[str, Any] = OrderedDict([
            ("type", "openai"),
            ("base_url", "${LLM_ROUTER_OPENAI_UPSTREAM_URL}"),
            ("auth", OrderedDict([
                ("type", "header_key"),
                ("header", "api-key"),
                ("value", "${LLM_ROUTER_OPENAI_API_KEY}"),
            ])),
            ("models", openai_list),
        ])
        # Only include deployments for models in this provider
        relevant_deployments = OrderedDict(
            (k, v) for k, v in deployments.items() if k in openai_models
        )
        if relevant_deployments:
            provider_data["deployments"] = relevant_deployments
        providers["azure-openai"] = provider_data

    if vertex_models:
        providers["google-vertex"] = OrderedDict([
            ("type", "vertex"),
            ("base_url", "${LLM_ROUTER_VERTEX_UPSTREAM_URL}"),
            ("auth", OrderedDict([
                ("type", "bearer"),
                ("value", "${LLM_ROUTER_VERTEX_API_KEY}"),
            ])),
            ("models", sorted(vertex_models)),
        ])

    return providers


def build_routing_rules(rules: dict[str, Any]) -> list[dict[str, Any]]:
    """③④⑤⑥⑦ Convert force_overrides, cron_routes, keywords, length_fallback → rules list."""
    result: list[dict[str, Any]] = []

    # ③ force_overrides → pattern rules
    for fo in rules.get("force_overrides", []):
        result.append(OrderedDict([
            ("name", f"force-{fo['pattern'].replace(':', '-')}"),
            ("match", OrderedDict([
                ("type", "pattern"),
                ("field", "system_prompt"),
                ("pattern", fo["pattern"]),
            ])),
            ("model", fo["model"]),
        ]))

    # ④ cron_routes → pattern rules with extract
    cron_routes = rules.get("cron_routes", {})
    for cron_name, model in cron_routes.items():
        result.append(OrderedDict([
            ("name", f"cron-{cron_name}"),
            ("match", OrderedDict([
                ("type", "pattern"),
                ("field", "system_prompt"),
                ("pattern", cron_name),
                ("extract", r"agent:\w+:(cron|chat):[^\s]+"),
            ])),
            ("model", model),
        ]))

    # ⑤⑥ keywords → split by tier into keyword rules
    keywords = rules.get("keywords", {})
    if keywords:
        tier_groups: dict[str, dict[str, Any]] = {}
        for kw, data in keywords.items():
            tier = data.get("tier", "T2")
            if tier not in tier_groups:
                tier_groups[tier] = {}
            tier_groups[tier][kw] = OrderedDict([
                ("weight_a", data.get("opus_weight", 0.5)),
                ("weight_b", data.get("sonnet_weight", 0.5)),
                ("source", data.get("source", "manual")),
            ])

        # Determine model per tier
        tier_model_map = {
            "T1": "claude-opus-4-6",
            "T2": "claude-sonnet-4-6",
            "T3": "claude-sonnet-4-6",
        }
        tier_fallback_map: dict[str, str | None] = {
            "T1": "claude-sonnet-4-6",
            "T2": None,
            "T3": None,
        }

        for tier in sorted(tier_groups.keys()):
            kws = tier_groups[tier]
            rule: dict[str, Any] = OrderedDict([
                ("name", f"keyword-tier-{tier.lower()}"),
                ("match", OrderedDict([
                    ("type", "keyword"),
                    ("field", "all_text"),
                    ("keywords", kws),
                    ("threshold", rules.get("keyword_score_threshold", 0.15)),
                ])),
                ("model", tier_model_map.get(tier, "claude-sonnet-4-6")),
                ("tier", tier),
            ])
            fb = tier_fallback_map.get(tier)
            if fb:
                rule["fallback_model"] = fb
            result.append(rule)

    # ⑦ length_fallback → 2 length rules
    lf = rules.get("length_fallback", {})
    if lf:
        threshold = lf.get("threshold", 2000)
        result.append(OrderedDict([
            ("name", "short-context"),
            ("match", OrderedDict([
                ("type", "length"),
                ("field", "user_message"),
                ("max_chars", threshold - 1),
            ])),
            ("model", lf.get("short_model", "claude-sonnet-4-6")),
        ]))
        result.append(OrderedDict([
            ("name", "long-context"),
            ("match", OrderedDict([
                ("type", "length"),
                ("field", "user_message"),
                ("min_chars", threshold),
            ])),
            ("model", lf.get("long_model", "claude-opus-4-6")),
        ]))

    return result


def build_metadata(rules: dict[str, Any]) -> dict[str, Any]:
    """⑧ inject_metadata → metadata section."""
    im = rules.get("inject_metadata", {})
    enabled = im.get("enabled", True)

    field_map = [
        ("include_timestamp", "timestamp"),
        ("include_model", "model"),
        ("include_elapsed", "elapsed"),
        ("include_tokens", "tokens"),
        ("include_reason", "stop_reason"),
    ]
    fields = [name for key, name in field_map if im.get(key, False)]

    return OrderedDict([
        ("enabled", enabled),
        ("format", "footer"),
        ("fields", fields),
    ])


def build_learning(rules: dict[str, Any]) -> dict[str, Any]:
    """⑨ auto_learn → learning section."""
    al = rules.get("auto_learn", {})
    return OrderedDict([
        ("enabled", al.get("enabled", False)),
        ("alpha", al.get("alpha", 0.1)),
        ("min_weight", al.get("min_weight", 0.1)),
        ("max_weight", al.get("max_weight", 0.9)),
        ("max_keywords_per_update", al.get("max_keywords_per_update", 5)),
        ("protect_manual", al.get("protect_manual", True)),
    ])


def build_fallback(rules: dict[str, Any]) -> dict[str, Any]:
    """⑩ Build fallback section (remove migrated fields)."""
    fb = rules.get("fallback", {})
    result: dict[str, Any] = OrderedDict([
        ("enabled", fb.get("enabled", True)),
        ("default_tier", fb.get("default_tier", "T1")),
        ("tiers", fb.get("tiers", {})),
        ("max_retries", fb.get("max_retries", 8)),
        ("circuit_breaker", OrderedDict([
            ("failure_threshold", 3),
            ("recovery_seconds", 300),
        ])),
        ("retry_on_status", fb.get("retry_on_status", [429, 500, 502, 503])),
    ])
    # migrated fields NOT included: openai_models, vertex_models, deployments
    return result


def migrate(input_path: str, output_path: str) -> None:
    """Convert rules.json → config.yaml."""
    raw = Path(input_path).read_text(encoding="utf-8")
    rules = json.loads(raw)

    # ① default_model, global_override → routing section
    routing = OrderedDict([
        ("default_model", rules.get("default_model", "claude-sonnet-4-6")),
        ("global_override", rules.get("global_override")),
        ("rules", build_routing_rules(rules)),
    ])

    # ② providers
    providers = build_providers(rules)

    # Build full config
    config = OrderedDict([
        ("version", 1),
        ("server", OrderedDict([
            ("host", "0.0.0.0"),
            ("port", 8100),
            ("log_level", "warning"),
        ])),
        ("providers", providers),
        ("routing", routing),
        ("fallback", build_fallback(rules)),
        ("metadata", build_metadata(rules)),
        ("stats", OrderedDict([
            ("enabled", True),
            ("path", "./stats.jsonl"),
            ("max_bytes", 10_485_760),
        ])),
        ("learning", build_learning(rules)),
    ])

    from ruamel.yaml import YAML  # type: ignore[import-untyped]
    yml: Any = YAML()
    yml.default_flow_style = False
    # Preserve OrderedDict ordering
    def _represent_ordereddict(dumper: Any, data: Any) -> Any:
        return dumper.represent_mapping("tag:yaml.org,2002:map", data)

    yml.Representer.add_representer(OrderedDict, _represent_ordereddict)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yml.dump(config, f)

    print(f"Migrated {input_path} → {output_path}")
    print(f"  providers: {len(providers)}")
    print(f"  routing rules: {len(routing['rules'])}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate rules.json → config.yaml"
    )
    parser.add_argument(
        "--input", required=True, help="Path to rules.json"
    )
    parser.add_argument(
        "--output", required=True, help="Output path for config.yaml"
    )
    args = parser.parse_args()
    migrate(args.input, args.output)


if __name__ == "__main__":
    main()
