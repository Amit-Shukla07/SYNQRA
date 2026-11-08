"""Main CLI entrypoint for SYNQRA (Phase 9.6).

Provides unified command dispatch for single-shot agent execution, interactive chat,
system diagnostics, and MCP server launch.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

import anyio
import click

from synqra.cli.chat import run_chat_session
from synqra.cli.commands import display_status, display_version, run_agent_request
from synqra.core.config import get_settings


@click.group(
    invoke_without_command=True,
    help="SYNQRA – The Universal AI Integration Agent and MCP Server.",
)
@click.option("--version", "-v", "show_version", is_flag=True, help="Show SYNQRA version.")
@click.pass_context
def app(ctx: click.Context, show_version: bool) -> None:
    """SYNQRA command-line interface entry point."""
    if show_version:
        display_version()
        ctx.exit(0)
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@app.command("version", help="Show application version and environment.")
def version_cmd() -> None:
    """Print version details."""
    display_version()


@app.command("status", help="Show system status and integration readiness.")
def status_cmd() -> None:
    """Display comprehensive system status."""
    anyio.run(display_status)


@app.command("run", help="Execute a single natural-language agent workflow.")
@click.argument("request_text", type=str)
@click.option("--thread-id", "-t", default=None, help="Optional thread ID for execution.")
def run_cmd(request_text: str, thread_id: str | None) -> None:
    """Run single-shot agent request."""
    anyio.run(run_agent_request, request_text, thread_id)


@app.command("chat", help="Start an interactive multi-turn terminal chat session.")
def chat_cmd() -> None:
    """Start interactive chat."""
    anyio.run(run_chat_session)


@app.command(
    "server",
    help="Launch the SYNQRA FastMCP server with Streamable HTTP transport.",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.option("--host", type=str, default=None, help="Host to bind (e.g. 127.0.0.1).")
@click.option("--port", type=int, default=None, help="Port to bind (e.g. 8000).")
@click.option("--debug", is_flag=True, default=False, help="Enable debug mode.")
def server_cmd(host: str | None, port: int | None, debug: bool) -> None:
    """Launch the FastMCP server."""
    from synqra.server.app import run_server

    cfg = get_settings()
    overrides: dict[str, Any] = {}
    if host is not None:
        overrides["host"] = host
    if port is not None:
        overrides["port"] = port
    if debug:
        overrides["debug"] = True

    settings = cfg.model_copy(update=overrides) if overrides else cfg
    run_server(settings)


def main(args: Sequence[str] | None = None) -> int:
    """Entry point for console script invocation."""
    try:
        app(args=list(args) if args is not None else None, standalone_mode=False)
        return 0
    except click.ClickException as e:
        e.show()
        return e.exit_code
    except click.exceptions.Exit as e:
        return e.exit_code
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
