"""Terminal display and rich presentation formatting for SYNQRA CLI (Phase 9.6)."""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def render_banner() -> None:
    """Render the SYNQRA branded terminal banner."""
    banner_text = (
        "[bold cyan]"
        "╔══════════════════════════════════════════════╗\n"
        "║                  SYNQRA                      ║\n"
        "║          Universal AI Integration Agent      ║\n"
        "╚══════════════════════════════════════════════╝"
        "[/bold cyan]"
    )
    console.print(banner_text)


def render_status_table(
    app_version: str,
    environment: str,
    transport: str,
    checkpointer_type: str,
    integrations_status: dict[str, Any],
) -> None:
    """Render a clean status dashboard showing server and integration readiness."""
    table = Table(title="[bold green]SYNQRA System Status[/bold green]", show_header=True)
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Details", style="dim")

    table.add_row("Core Server", "[green]Online[/green]", f"v{app_version} ({environment})")
    table.add_row("Transport", "[green]Active[/green]", transport)
    table.add_row("Checkpointer", "[green]Active[/green]", f"Backend: {checkpointer_type}")

    for name, status in integrations_status.items():
        disp = getattr(status, "display_name", name.replace("_", " ").title())
        is_cfg = getattr(status, "is_configured", False)
        is_conn = getattr(status, "is_connected", False)
        st = getattr(status, "state", None)
        st_val = st.value if hasattr(st, "value") else str(st or "")
        err_msg = getattr(status, "error_message", None)

        if st_val == "connected" or (is_cfg and is_conn):
            state_str = "[green]✓ Connected[/green]"
            details = "Active & verified"
        elif st_val == "degraded":
            state_str = "[yellow]⚠ Degraded[/yellow]"
            details = err_msg or "Configured; connectivity degraded"
        elif st_val == "error":
            state_str = "[red]✗ Error[/red]"
            details = err_msg or "Service connection error"
        elif is_cfg:
            state_str = "[cyan]○ Configured[/cyan]"
            details = "Credentials present (not connected)"
        else:
            state_str = "[yellow]○ Unconfigured[/yellow]"
            details = "Credentials pending in environment"

        table.add_row(f"Integration: {disp}", state_str, details)

    console.print(table)


def render_integrations_summary(integrations: list[dict[str, Any]]) -> None:
    """Render quick checklist of available integrations."""
    console.print("\n[bold]Configured integrations:[/bold]")
    for item in integrations:
        name = item.get("name", "Unknown")
        configured = item.get("configured", False)
        symbol = "[cyan]✓[/cyan]" if configured else "[dim]○[/dim]"
        status_text = "[cyan]Configured[/cyan]" if configured else "[dim]Not configured[/dim]"
        console.print(f" {symbol} {name:<20} {status_text}")
    console.print("")


def render_progress_step(index: int, description: str, tool_name: str | None = None) -> None:
    """Display an incremental step in agent plan execution."""
    tool_suffix = f" ([cyan]{tool_name}[/cyan])" if tool_name else ""
    console.print(f"[bold blue][{index}][/bold blue] {description}{tool_suffix}...")


def render_approval_notice(
    approval_id: str,
    tool_name: str,
    workflow_id: str,
    thread_id: str,
    permission_level: str,
    arguments: dict[str, Any],
    reason: str | None = None,
) -> None:
    """Render high-visibility security panel for human approval interruptions."""
    args_pretty = json.dumps(arguments, indent=2)
    body = (
        f"[bold yellow]⚠️ Human Authorization Required[/bold yellow]\n\n"
        f"This workflow requested a protected [bold red]{permission_level.upper()}[/bold red] "
        f"operation that cannot proceed without explicit approval.\n\n"
        f"• [bold]Approval Request ID:[/bold] [cyan]{approval_id}[/cyan]\n"
        f"• [bold]Tool Name:[/bold]           [bold]{tool_name}[/bold]\n"
        f"• [bold]Workflow ID:[/bold]         {workflow_id}\n"
        f"• [bold]Thread ID:[/bold]           {thread_id}\n"
        f"• [bold]Action Purpose:[/bold]      {reason or 'N/A'}\n\n"
        f"[bold]Requested Arguments:[/bold]\n"
        f"[dim]{args_pretty}[/dim]\n\n"
        f"[bold green]To authorize this operation:[/bold green]\n"
        f"Submit an approval decision to the SYNQRA Approval API:\n"
        f"  [cyan]POST /api/approvals/{approval_id}/approve[/cyan]\n"
        f"  Header: [cyan]Authorization: Bearer <your-operator-token>[/cyan]"
    )
    console.print(
        Panel(
            body,
            title="[bold yellow]SECURITY INTERRUPT (Human-in-the-Loop)[/bold yellow]",
            border_style="yellow",
        )
    )


def render_final_response(content: str) -> None:
    """Display the final synthesized response from SYNQRA."""
    console.print(f"\n[bold green]SYNQRA >[/bold green] {content}\n")


def render_error(message: str) -> None:
    """Display a formatted error message without exposing secrets."""
    console.print(f"[bold red]Error:[/bold red] {message}")
