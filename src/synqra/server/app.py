"""Application entry point for the SYNQRA MCP server.

Loads configuration, configures logging, and launches the FastMCP server
using the Streamable HTTP transport.
"""

import argparse
import logging
import sys
from collections.abc import Sequence

from synqra.core.config import Settings, get_settings
from synqra.server.mcp_server import create_mcp_server

logger = logging.getLogger("synqra")


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the server runner."""
    parser = argparse.ArgumentParser(
        prog="synqra",
        description="SYNQRA – The Universal MCP Integration Server",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host interface to bind (defaults to SYNQRA_HOST or 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port number to bind (defaults to SYNQRA_PORT or 8000)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=None,
        help="Enable debug mode and verbose logging",
    )
    return parser.parse_args(args)


def setup_logging(debug: bool = False) -> None:
    """Configure structured logging for the server process."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run_server(settings: Settings | None = None) -> None:
    """Initialize and run the FastMCP server with Streamable HTTP transport.

    Args:
        settings: Application settings. If None, settings are fetched via get_settings().
    """
    cfg = settings or get_settings()
    setup_logging(debug=cfg.debug)

    logger.info(
        "Starting %s v%s (environment: %s, transport: %s, endpoint: http://%s:%s%s)",
        cfg.app_name,
        cfg.app_version,
        cfg.environment,
        cfg.transport,
        cfg.host,
        cfg.port,
        cfg.mcp_path,
    )

    server = create_mcp_server(cfg)

    # 1. Pre-flight check: fail fast if PostgreSQL checkpointer is configured but unavailable
    if cfg.checkpointer_type == "postgres":
        logger.info("Verifying PostgreSQL checkpointer connection and schema readiness...")

        async def _startup_preflight() -> None:
            from synqra.agents.checkpointer import get_checkpointer

            await get_checkpointer("postgres", database_url=cfg.database_url, timeout=2.0)

        import anyio

        anyio.run(_startup_preflight)
        logger.info("PostgreSQL checkpointer verified and ready.")

    try:
        server.run(
            transport=cfg.transport,
            host=cfg.host,
            port=cfg.port,
            path=cfg.mcp_path,
            stateless_http=cfg.stateless_http,
            log_level="debug" if cfg.debug else "info",
        )
    except KeyboardInterrupt:
        logger.info("Server shutdown requested via keyboard interrupt.")
    except Exception as exc:
        logger.error("Failed to run MCP server: %s", exc, exc_info=cfg.debug)
        raise
    finally:
        logger.info("Executing server shutdown cleanup...")

        async def _shutdown_cleanup() -> None:
            import contextlib

            from synqra.agents.checkpointer import close_checkpointer
            from synqra.database.session import dispose_async_engine

            with contextlib.suppress(Exception):
                await close_checkpointer()
            with contextlib.suppress(Exception):
                await dispose_async_engine()

        import anyio

        try:
            anyio.run(_shutdown_cleanup)
        except Exception as cleanup_err:
            logger.warning("Error during server shutdown cleanup: %s", cleanup_err)


def main(args: Sequence[str] | None = None) -> int:
    """Entry point for CLI execution (`synqra`).

    Args:
        args: Command line arguments. If None, sys.argv[1:] is used.

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    try:
        cli_args = parse_args(args)
        base_settings = get_settings()

        # Build overrides from CLI arguments if explicitly provided
        overrides: dict[str, str | int | bool] = {}
        if cli_args.host is not None:
            overrides["host"] = cli_args.host
        if cli_args.port is not None:
            overrides["port"] = cli_args.port
        if cli_args.debug is not None and cli_args.debug:
            overrides["debug"] = True

        settings = base_settings.model_copy(update=overrides) if overrides else base_settings

        run_server(settings)
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as err:
        sys.stderr.write(f"Error: {err}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
