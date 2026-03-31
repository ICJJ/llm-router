"""CLI entry point for llm-router: serve and validate subcommands."""
from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="llm-router",
        description="LLM Router — intelligent model routing proxy",
    )
    sub = parser.add_subparsers(dest="command")

    # ── serve ──
    serve_p = sub.add_parser("serve", help="Start the router server")
    serve_p.add_argument("--config", default="config.yaml", help="YAML config path")
    serve_p.add_argument("--host", default="0.0.0.0", help="Bind host")
    serve_p.add_argument("--port", type=int, default=8100, help="Bind port")
    serve_p.add_argument(
        "--log-level", default="warning", choices=["debug", "info", "warning", "error"],
    )

    # ── validate ──
    validate_p = sub.add_parser("validate", help="Validate a YAML config file")
    validate_p.add_argument("--config", required=True, help="YAML config path")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "serve":
        _serve(args)
    elif args.command == "validate":
        _validate(args)


def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    os.environ["LLM_ROUTER_RULES_PATH"] = args.config
    os.environ["LLM_ROUTER_PORT"] = str(args.port)
    os.environ["LLM_ROUTER_LOG_LEVEL"] = args.log_level
    uvicorn.run(
        "llm_router.main:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


def _validate(args: argparse.Namespace) -> None:
    from .config import load_config

    try:
        load_config(args.config)
    except Exception as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"Configuration valid: {args.config}")


if __name__ == "__main__":
    main()
