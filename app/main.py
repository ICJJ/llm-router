"""FastAPI application factory and route registration."""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request

from . import stats
from .config import get_settings, init_config, get_config
from .proxy import close_client, handle_messages

logger = logging.getLogger("llm-router")

_start_time: float = 0.0
_request_count: int = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _start_time
    settings = get_settings()
    config_path = Path(settings.rules_path)
    if not config_path.exists():
        msg = (
            f"YAML config 文件未提供: {config_path.resolve()}\n"
            f"请创建 llm-router.yaml，或设置环境变量 LLM_ROUTER_RULES_PATH 指向有效的 YAML 配置文件。\n"
            f"参考 config.example.yaml 创建配置。"
        )
        logger.critical(msg)
        raise SystemExit(msg)
    init_config(settings.rules_path)
    stats.init(settings.stats_path, settings.stats_max_bytes)
    _start_time = time.monotonic()
    yield
    await close_client()


def create_app() -> FastAPI:
    application = FastAPI(title="LLM Router", lifespan=lifespan)

    @application.post("/v1/messages")
    async def messages(request: Request):  # pyright: ignore[reportUnusedFunction]
        global _request_count
        _request_count += 1
        response = await handle_messages(request)
        return response

    @application.post("/v1/chat/completions")
    @application.post("/v1/completions")
    @application.post("/chat/completions")
    @application.post("/completions")
    async def chat_completions(request: Request):  # pyright: ignore[reportUnusedFunction]
        global _request_count
        _request_count += 1
        from .proxy import handle_chat_completions
        response = await handle_chat_completions(request)
        return response

    @application.get("/health")
    async def health() -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        from . import fallback as fb
        uptime = time.monotonic() - _start_time if _start_time else 0
        return {
            "status": "ok",
            "uptime_seconds": int(uptime),
            "rules_loaded": True,
            "total_requests": _request_count,
            "model_health": fb.get_health_status(),
            "global_override": get_config().routing.global_override,
        }

    return application


app = create_app()
