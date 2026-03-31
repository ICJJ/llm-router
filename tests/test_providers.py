"""Tests for llm_router.providers – Phase 2 Provider Registry."""
from __future__ import annotations

import pytest

import llm_router.providers as prov_mod
from llm_router.config import AuthConfig, ProviderConfig
from llm_router.providers import (
    ProviderRegistry,
    ResolvedProvider,
    get_registry,
    init_registry,
)

# ── Fixtures ────────────────────────────────────────────────────────

ANTHROPIC_CFG = ProviderConfig(
    type="anthropic",
    base_url="https://api.anthropic.com",
    auth=AuthConfig(type="bearer", value="sk-ant-123"),
    models=["claude-opus-4-6", "claude-sonnet-4-6"],
)

OPENAI_CFG = ProviderConfig(
    type="openai",
    base_url="https://api.openai.com",
    auth=AuthConfig(type="header_key", header="api-key", value="sk-oai-456"),
    models=["gpt-4o", "gpt-5.4"],
    deployments={"gpt-5.4": "dep-gpt54"},
)

VERTEX_CFG = ProviderConfig(
    type="vertex",
    base_url="https://vertex.googleapis.com",
    auth=AuthConfig(type="bearer", value="ya29.token"),
    models=["gemini-2.5-pro"],
)


@pytest.fixture()
def registry() -> ProviderRegistry:
    """Standard 3-provider registry."""
    return ProviderRegistry(
        providers={
            "anthropic": ANTHROPIC_CFG,
            "openai": OPENAI_CFG,
            "google-vertex": VERTEX_CFG,
        },
    )


@pytest.fixture()
def registry_with_default() -> ProviderRegistry:
    """Registry with a default_provider for unknown model fallback."""
    return ProviderRegistry(
        providers={
            "anthropic": ANTHROPIC_CFG,
            "openai": OPENAI_CFG,
        },
        default_provider="anthropic",
    )


@pytest.fixture(autouse=True)
def _reset_global_registry():
    """Reset module-level _registry before and after each test."""
    prov_mod._registry = None
    yield
    prov_mod._registry = None


# ── 1. Construction ─────────────────────────────────────────────────


class TestProviderRegistryConstruction:
    def test_distinct_models_build_correct_model_map(self, registry: ProviderRegistry):
        expected = {"claude-opus-4-6", "claude-sonnet-4-6", "gpt-4o", "gpt-5.4", "gemini-2.5-pro"}
        assert registry.all_models == expected

    def test_duplicate_model_raises_value_error(self):
        dup = ProviderConfig(
            type="openai",
            base_url="https://dup.example.com",
            auth=AuthConfig(type="none"),
            models=["claude-opus-4-6"],
        )
        with pytest.raises(ValueError, match="claude-opus-4-6"):
            ProviderRegistry(providers={"anthropic": ANTHROPIC_CFG, "dup": dup})

    def test_empty_providers_dict(self):
        reg = ProviderRegistry(providers={})
        assert reg.all_models == set()


# ── 2. resolve() ────────────────────────────────────────────────────


class TestResolve:
    def test_known_model_returns_correct_resolved(self, registry: ProviderRegistry):
        r = registry.resolve("claude-opus-4-6")
        assert isinstance(r, ResolvedProvider)
        assert r.name == "anthropic"
        assert r.provider_type == "anthropic"
        assert r.base_url == "https://api.anthropic.com"

    def test_model_with_custom_deployment(self, registry: ProviderRegistry):
        r = registry.resolve("gpt-5.4")
        assert r.deployment == "dep-gpt54"

    def test_model_without_deployment_uses_model_name(self, registry: ProviderRegistry):
        r = registry.resolve("gpt-4o")
        assert r.deployment == "gpt-4o"

    def test_unknown_model_no_default_raises_key_error(self, registry: ProviderRegistry):
        with pytest.raises(KeyError, match="unknown-model"):
            registry.resolve("unknown-model")

    def test_unknown_model_with_default_resolves_to_default(
        self, registry_with_default: ProviderRegistry
    ):
        r = registry_with_default.resolve("some-new-claude-model")
        assert r.name == "anthropic"
        assert r.provider_type == "anthropic"

    def test_resolved_provider_extra_headers(self):
        cfg = ProviderConfig(
            type="anthropic",
            base_url="https://example.com",
            auth=AuthConfig(type="none"),
            models=["m1"],
            headers={"X-Custom": "val"},
        )
        reg = ProviderRegistry(providers={"p": cfg})
        r = reg.resolve("m1")
        assert r.extra_headers == {"X-Custom": "val"}


# ── 3. get_request_url() ───────────────────────────────────────────


class TestGetRequestUrl:
    def test_openai_model_url(self, registry: ProviderRegistry):
        url = registry.get_request_url("gpt-4o")
        assert url == (
            "https://api.openai.com/openai/deployments/gpt-4o"
            "/chat/completions?api-version=2024-10-21"
        )

    def test_openai_model_with_deployment_url(self, registry: ProviderRegistry):
        url = registry.get_request_url("gpt-5.4")
        assert url == (
            "https://api.openai.com/openai/deployments/dep-gpt54"
            "/chat/completions?api-version=2024-10-21"
        )

    def test_vertex_model_url(self, registry: ProviderRegistry):
        url = registry.get_request_url("gemini-2.5-pro")
        assert url == "https://vertex.googleapis.com/v1/chat/completions"

    def test_anthropic_model_url(self, registry: ProviderRegistry):
        url = registry.get_request_url("claude-opus-4-6")
        assert url == "https://api.anthropic.com/v1/messages"


# ── 4. get_request_headers() ───────────────────────────────────────


class TestGetRequestHeaders:
    def test_header_key_auth(self, registry: ProviderRegistry):
        hdrs = registry.get_request_headers("gpt-4o")
        assert hdrs["api-key"] == "sk-oai-456"

    def test_bearer_auth(self, registry: ProviderRegistry):
        hdrs = registry.get_request_headers("claude-opus-4-6")
        assert hdrs["Authorization"] == "Bearer sk-ant-123"

    def test_none_auth_no_auth_headers(self):
        cfg = ProviderConfig(
            type="vertex",
            base_url="https://example.com",
            auth=AuthConfig(type="none"),
            models=["m1"],
        )
        reg = ProviderRegistry(providers={"p": cfg})
        hdrs = reg.get_request_headers("m1")
        assert "Authorization" not in hdrs
        # default header name from AuthConfig is "Authorization" – should not appear
        assert hdrs.get("Authorization") is None

    def test_anthropic_model_has_anthropic_version(self, registry: ProviderRegistry):
        hdrs = registry.get_request_headers("claude-opus-4-6")
        assert hdrs["anthropic-version"] == "2023-06-01"

    def test_anthropic_custom_version(self, registry: ProviderRegistry):
        hdrs = registry.get_request_headers("claude-opus-4-6", anthropic_version="2024-01-01")
        assert hdrs["anthropic-version"] == "2024-01-01"

    def test_anthropic_with_beta(self, registry: ProviderRegistry):
        hdrs = registry.get_request_headers(
            "claude-sonnet-4-6", anthropic_beta="max-tokens-3-5-sonnet-2024-07-15"
        )
        assert hdrs["anthropic-version"] == "2023-06-01"
        assert hdrs["anthropic-beta"] == "max-tokens-3-5-sonnet-2024-07-15"

    def test_openai_no_anthropic_headers(self, registry: ProviderRegistry):
        hdrs = registry.get_request_headers("gpt-4o")
        assert "anthropic-version" not in hdrs
        assert "anthropic-beta" not in hdrs

    def test_vertex_no_anthropic_headers(self, registry: ProviderRegistry):
        hdrs = registry.get_request_headers("gemini-2.5-pro")
        assert "anthropic-version" not in hdrs

    def test_content_type_always_present(self, registry: ProviderRegistry):
        for model in ["claude-opus-4-6", "gpt-4o", "gemini-2.5-pro"]:
            hdrs = registry.get_request_headers(model)
            assert hdrs["Content-Type"] == "application/json"

    def test_extra_headers_merged(self):
        cfg = ProviderConfig(
            type="anthropic",
            base_url="https://example.com",
            auth=AuthConfig(type="none"),
            models=["m1"],
            headers={"X-Trace-Id": "abc123"},
        )
        reg = ProviderRegistry(providers={"p": cfg})
        hdrs = reg.get_request_headers("m1")
        assert hdrs["X-Trace-Id"] == "abc123"
        assert hdrs["Content-Type"] == "application/json"


# ── 5. get_provider_type() / is_known_model() / all_models ──────────


class TestHelpers:
    def test_get_provider_type(self, registry: ProviderRegistry):
        assert registry.get_provider_type("claude-opus-4-6") == "anthropic"
        assert registry.get_provider_type("gpt-4o") == "openai"
        assert registry.get_provider_type("gemini-2.5-pro") == "vertex"

    def test_is_known_model_true(self, registry: ProviderRegistry):
        assert registry.is_known_model("claude-opus-4-6") is True

    def test_is_known_model_false_no_default(self, registry: ProviderRegistry):
        assert registry.is_known_model("nonexistent") is False

    def test_is_known_model_unknown_but_has_default(
        self, registry_with_default: ProviderRegistry
    ):
        assert registry_with_default.is_known_model("any-model") is True

    def test_all_models_returns_set(self, registry: ProviderRegistry):
        models = registry.all_models
        assert isinstance(models, set)
        assert len(models) == 5


# ── 6. Module-level functions ───────────────────────────────────────


class TestModuleLevelFunctions:
    def test_init_registry_and_get_registry_returns_same_instance(self):
        providers = {"anthropic": ANTHROPIC_CFG}
        reg = init_registry(providers, default_provider="anthropic")
        assert get_registry() is reg

    def test_init_registry_returns_provider_registry(self):
        reg = init_registry({"openai": OPENAI_CFG})
        assert isinstance(reg, ProviderRegistry)
        assert "gpt-4o" in reg.all_models

    def test_get_registry_before_init_auto_builds(self, monkeypatch):
        """get_registry() with no prior init falls back to config/legacy build."""
        fake_providers = {
            "test-prov": ProviderConfig(
                type="anthropic",
                base_url="https://fake.example.com",
                auth=AuthConfig(type="none"),
                models=["fake-model"],
            ),
        }

        class FakeConfig:
            providers = fake_providers

        monkeypatch.setattr("llm_router.config.get_config", lambda: FakeConfig())
        reg = get_registry()
        assert isinstance(reg, ProviderRegistry)
        assert "fake-model" in reg.all_models

    def test_get_registry_falls_back_to_legacy_when_no_yaml_providers(self, monkeypatch):
        """When config.providers is empty, falls back to _build_legacy_registry."""

        class FakeConfig:
            providers = {}

        sentinel_registry = ProviderRegistry(providers={"v": VERTEX_CFG})
        monkeypatch.setattr("llm_router.config.get_config", lambda: FakeConfig())
        monkeypatch.setattr(prov_mod, "_build_legacy_registry", lambda: sentinel_registry)
        reg = get_registry()
        assert reg is sentinel_registry
