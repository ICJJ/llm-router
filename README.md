# LLM Router

Intelligent model routing proxy for LLM APIs. Sits between your application and multiple LLM providers (OpenAI/Azure, Gemini/Vertex), automatically selecting the best model based on configurable rules.

## Features

- **Dual Protocol** — Anthropic Messages API + OpenAI Chat Completions API, both proxied through a single service.
- **Declarative Routing** — 6-level classifier: global override, model hint, pattern matching, keyword scoring, message length, default with optional performance routing. First match wins.
- **Multi-Provider Support** — OpenAI, Azure OpenAI, Google Vertex AI with unified proxy interface and format translation.
- **Tiered Fallback** — Circuit breaker per model with configurable failure thresholds and recovery. Automatic retry across T1/T2 fallback tiers.
- **Format Translation** — Anthropic ↔ OpenAI request/response translation, AMD response normalization (3 formats), OpenAI SSE → Anthropic SSE streaming.
- **SSE Burst Mode** — Synthesizes Anthropic SSE event streams from non-streaming upstream responses (Gemini via AMD unified endpoint).
- **Auto-Learning** — Optional keyword weight adjustment based on routing outcomes (disabled by default).
- **Response Metadata** — Injects timestamp, model name, latency, token counts into responses.
- **YAML Configuration** — Hot-reloading config with `${ENV_VAR}` substitution and Pydantic v2 validation.
- **Chat Commands** — `/route` commands for runtime rule inspection and modification via Feishu.
- **Stats Logging** — JSONL request logging with automatic file rotation.

## Quick Start

```bash
# Install
pip install -e .

# Configure
cp config.example.yaml llm-router.yaml
# Edit llm-router.yaml with your providers and API keys
# In production (Docker): /home/node/.openclaw/config/llm-router.yaml

# Run
uvicorn app.main:app --host 0.0.0.0 --port 8100
```

Point your application to `http://localhost:8100/v1/messages` (Anthropic format) or `http://localhost:8100/v1/chat/completions` (OpenAI format).

## Configuration

See [config.example.yaml](config.example.yaml) for a fully documented example. Key sections:

- **providers** — Define LLM backends with auth, base URLs, and model lists
- **routing.rules** — Declarative routing rules (pattern / keyword / length)
- **routing.performance** — Optional latency/throughput-based model selection
- **fallback** — Tier chains, circuit breaker settings, retry policies
- **metadata** — Response metadata injection (timestamp, model, latency, tokens)
- **learning** — Auto keyword weight tuning (disabled by default)

Environment variables are supported via `${VAR_NAME}` syntax in YAML values.

## Architecture

```
Client ──► LLM Router (:8100) ──► Provider A (OpenAI/Azure)
                                ──► Provider B (Gemini/Vertex)
```

```
app/
├── main.py          # FastAPI app factory, lifespan, routes
├── config.py        # YAML config loader, Pydantic v2 models, hot-reload
├── router.py        # Declarative rule engine (pattern/keyword/length/model_hint)
├── proxy.py         # Async HTTP proxy with SSE streaming, burst mode, fallback
├── providers.py     # Provider registry, model → upstream URL resolution
├── fallback.py      # Circuit breaker, tiered fallback chain
├── commands.py      # /route chat command parser and executor
├── translator.py    # Anthropic ↔ OpenAI format translation
├── metadata.py      # Response metadata injection
├── stats.py         # JSONL stats logging with rotation
├── learner.py       # Auto keyword weight adjustment (disabled by default)
├── performance.py   # Performance tracker for latency/throughput routing
migrations/
└── json_to_yaml.py  # Legacy rules.json → config.yaml migration
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/messages` | POST | Anthropic Messages API format proxy |
| `/v1/chat/completions` | POST | OpenAI Chat Completions format proxy |
| `/v1/completions` | POST | Alias → `/v1/chat/completions` |
| `/chat/completions` | POST | Alias → `/v1/chat/completions` |
| `/completions` | POST | Alias → `/v1/chat/completions` |
| `/health` | GET | Health check with uptime, request count, model health, global override |

The `/v1/messages` endpoint accepts Anthropic-format requests. The `/v1/chat/completions` endpoint accepts OpenAI-format requests. Use `/route` commands in the user message for runtime configuration (e.g., `/route list`, `/route override gpt-5.4`).

## Testing

```bash
pip install pytest pytest-asyncio httpx
python -m pytest tests/ -q
```

283 tests covering routing logic, config loading, provider resolution, fallback chains, commands, SSE streaming, and format translation.

## Requirements

- Python ≥ 3.11
- FastAPI, httpx, Pydantic Settings, uvicorn, ruamel.yaml

## License

MIT
