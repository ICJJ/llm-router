# LLM Router

Intelligent model routing proxy for LLM APIs. Sits between your application and multiple LLM providers (Anthropic, OpenAI, Azure OpenAI, Google Vertex AI), automatically selecting the best model based on configurable rules.

## Features

- **Declarative Routing** — Pattern matching, keyword scoring, and message length rules. First match wins, evaluated top-to-bottom.
- **Multi-Provider Support** — Anthropic, OpenAI, Azure OpenAI, Google Vertex AI with unified proxy interface.
- **Tiered Fallback** — Circuit breaker per model with configurable failure thresholds and recovery. Automatic retry across T1/T2/T3 fallback tiers.
- **Auto-Learning** — Optional keyword weight adjustment based on routing outcomes.
- **Response Metadata** — Injects model name, latency, token counts into responses.
- **YAML Configuration** — Hot-reloading config with `${ENV_VAR}` substitution and Pydantic v2 validation.
- **Chat Commands** — `/route` commands for runtime rule inspection and modification.
- **Stats Logging** — JSONL request logging with automatic file rotation.

## Quick Start

```bash
# Install
pip install -e .

# Configure
cp config.example.yaml config.yaml
# Edit config.yaml with your providers and API keys

# Run
uvicorn llm_router.main:app --host 0.0.0.0 --port 8100
```

Point your application to `http://localhost:8100/v1/messages` instead of the provider's API directly.

## Configuration

See [config.example.yaml](config.example.yaml) for a fully documented example. Key sections:

- **providers** — Define LLM backends with auth, base URLs, and model lists
- **routing.rules** — Declarative routing rules (pattern / keyword / length)
- **fallback** — Tier chains, circuit breaker settings, retry policies
- **metadata** — Response metadata injection (model, latency, tokens)
- **learning** — Auto keyword weight tuning

Environment variables are supported via `${VAR_NAME}` syntax in YAML values.

## Architecture

```
Client ──► LLM Router (:8100) ──► Provider A (Anthropic)
                                ──► Provider B (OpenAI)
                                ──► Provider C (Vertex AI)
```

```
llm_router/
├── main.py         # FastAPI app factory, /v1/messages + /health endpoints
├── config.py       # YAML config loader, Pydantic v2 models, hot-reload
├── router.py       # Declarative rule engine (pattern/keyword/length)
├── proxy.py        # Async HTTP proxy with SSE streaming support
├── providers.py    # Provider registry, model→upstream resolution
├── fallback.py     # Circuit breaker, tiered fallback chain
├── commands.py     # /route chat command parser and executor
├── learner.py      # Auto keyword weight adjustment
├── metadata.py     # Response metadata injection
├── translator.py   # Request/response format translation
├── stats.py        # JSONL stats logging with rotation
migrations/
└── json_to_yaml.py # Legacy rules.json → config.yaml migration
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/messages` | POST | Proxy endpoint — routes request to selected model |
| `/health` | GET | Health check with uptime, request count, model status |

The `/v1/messages` endpoint accepts Anthropic-format requests. Use `/route` commands in the user message for runtime configuration (e.g., `/route status`, `/route override opus`).

## Testing

```bash
pip install pytest pytest-asyncio httpx
python -m pytest tests/ -q
```

283 tests covering routing logic, config loading, provider resolution, fallback chains, commands, and SSE streaming.

## Requirements

- Python ≥ 3.11
- FastAPI, httpx, Pydantic Settings, uvicorn, ruamel.yaml

## License

MIT
