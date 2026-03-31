"""Provider registry: model → upstream resolution."""
from __future__ import annotations

from dataclasses import dataclass

from .config import AuthConfig, ProviderConfig


@dataclass
class ResolvedProvider:
    """All info needed to make an upstream request for a given model."""
    name: str
    provider_type: str  # "anthropic" | "openai" | "vertex"
    base_url: str
    deployment: str
    auth: AuthConfig
    extra_headers: dict[str, str]


class ProviderRegistry:
    """Maps model names to provider configurations."""

    def __init__(
        self,
        providers: dict[str, ProviderConfig],
        default_provider: str | None = None,
    ) -> None:
        """
        Args:
            providers: name → ProviderConfig map
            default_provider: provider name to use when model not found in any
                provider's models list (for backward compat with Anthropic models
                that aren't explicitly listed)
        """
        self._providers = dict(providers)
        self._default_provider = default_provider
        self._model_map: dict[str, str] = {}
        for name, p in providers.items():
            for model in p.models:
                if model in self._model_map:
                    raise ValueError(
                        f"model '{model}' registered in both "
                        f"'{self._model_map[model]}' and '{name}'"
                    )
                self._model_map[model] = name

    def resolve(self, model: str) -> ResolvedProvider:
        """Resolve model to its provider config and deployment name."""
        name = self._model_map.get(model)
        if name is None:
            if self._default_provider and self._default_provider in self._providers:
                name = self._default_provider
            else:
                raise KeyError(f"model '{model}' not registered in any provider")
        p = self._providers[name]
        deployment = p.deployments.get(model, model)
        return ResolvedProvider(
            name=name,
            provider_type=p.type,
            base_url=p.base_url,
            deployment=deployment,
            auth=p.auth,
            extra_headers=dict(p.headers),
        )

    def get_request_url(self, model: str) -> str:
        """Build the full upstream URL for a model request."""
        r = self.resolve(model)
        if r.provider_type == "openai":
            return (
                f"{r.base_url}/openai/deployments/{r.deployment}"
                f"/chat/completions?api-version=2024-10-21"
            )
        if r.provider_type == "vertex":
            return f"{r.base_url}/v1/chat/completions"
        return f"{r.base_url}/v1/messages"

    def get_request_headers(
        self,
        model: str,
        anthropic_version: str = "2023-06-01",
        anthropic_beta: str | None = None,
    ) -> dict[str, str]:
        """Build upstream request headers including auth."""
        r = self.resolve(model)
        hdrs: dict[str, str] = {"Content-Type": "application/json"}
        hdrs.update(r.extra_headers)
        if r.auth.type == "header_key":
            hdrs[r.auth.header] = r.auth.value
        elif r.auth.type == "bearer":
            hdrs["Authorization"] = f"Bearer {r.auth.value}"
        if r.provider_type == "anthropic":
            hdrs["anthropic-version"] = anthropic_version
            if anthropic_beta:
                hdrs["anthropic-beta"] = anthropic_beta
        return hdrs

    def get_provider_type(self, model: str) -> str:
        """Return the provider type for a model."""
        return self.resolve(model).provider_type

    def is_known_model(self, model: str) -> bool:
        """Check if a model is registered or has a default provider."""
        return model in self._model_map or self._default_provider is not None

    @property
    def all_models(self) -> set[str]:
        """All explicitly registered model names."""
        return set(self._model_map.keys())


# ── Module-level registry store ──────────────────────────────────

_registry: ProviderRegistry | None = None


def init_registry(
    providers: dict[str, ProviderConfig],
    default_provider: str | None = None,
) -> ProviderRegistry:
    """Initialize the global provider registry."""
    global _registry
    _registry = ProviderRegistry(providers, default_provider)
    return _registry


def get_registry() -> ProviderRegistry:
    """Get the global registry, building from legacy settings if needed."""
    global _registry
    if _registry is not None:
        return _registry
    # Try new YAML config first
    from .config import get_config
    cfg = get_config()
    if cfg.providers:
        _registry = ProviderRegistry(cfg.providers)
        return _registry
    # Fallback: build from legacy Settings
    _registry = _build_legacy_registry()
    return _registry


def _build_legacy_registry() -> ProviderRegistry:
    """Build a minimal ProviderRegistry from Settings (no rules.json needed)."""
    from .config import get_settings

    settings = get_settings()
    providers: dict[str, ProviderConfig] = {}

    # Anthropic (default — catches any model not in specific providers)
    providers["anthropic"] = ProviderConfig(
        type="anthropic",
        base_url=settings.upstream_url,
        auth=AuthConfig(
            type="header_key",
            header="Ocp-Apim-Subscription-Key",
            value=settings.upstream_api_key,
        ),
    )

    return ProviderRegistry(providers, default_provider="anthropic")
