"""Command implementations for the SYNQRA CLI (Phase 9.6)."""

from __future__ import annotations

import logging
from typing import Any

from synqra.agents.graph import execute_agent_workflow
from synqra.agents.schemas import WorkflowStatus
from synqra.cli.display import (
    console,
    render_approval_notice,
    render_error,
    render_final_response,
    render_progress_step,
    render_status_table,
)
from synqra.core.config import Settings, get_settings
from synqra.integrations import default_registry

logger = logging.getLogger("synqra.cli.commands")


async def get_system_status() -> dict[str, Any]:
    """Gather complete diagnostic status for CLI status display."""
    settings = get_settings()
    statuses = await default_registry.check_health_all()
    return {
        "app_name": settings.app_name,
        "app_version": settings.app_version,
        "environment": settings.environment,
        "transport": settings.transport,
        "checkpointer_type": settings.checkpointer_type,
        "integrations_status": statuses,
    }


async def display_status() -> None:
    """CLI handler for `synqra status`."""
    status_data = await get_system_status()
    render_status_table(
        app_version=status_data["app_version"],
        environment=status_data["environment"],
        transport=status_data["transport"],
        checkpointer_type=status_data["checkpointer_type"],
        integrations_status=status_data["integrations_status"],
    )


def display_version(settings: Settings | None = None) -> None:
    """CLI handler for `synqra version`."""
    cfg = settings or get_settings()
    console.print(
        f"[bold cyan]{cfg.app_name}[/bold cyan] version [bold green]{cfg.app_version}[/bold green]"
    )
    console.print(f"Environment: [dim]{cfg.environment}[/dim]")
    console.print(f"Checkpointer: [dim]{cfg.checkpointer_type}[/dim]")


def format_agent_response(state: dict[str, Any]) -> dict[str, Any]:
    """Convert raw LangGraph workflow state into standardized Component C response structure."""
    workflow_id = state.get("workflow_id", "")
    thread_id = state.get("thread_id", "")
    raw_status = state.get("workflow_status", WorkflowStatus.PENDING)
    status_str = raw_status.value if hasattr(raw_status, "value") else str(raw_status)

    plan = state.get("plan")
    steps_summary = []
    if plan and hasattr(plan, "steps"):
        for step in plan.steps:
            steps_summary.append(
                {
                    "step_id": step.step_id,
                    "tool_name": step.tool_name,
                    "description": step.description,
                    "status": step.status.value
                    if hasattr(step.status, "value")
                    else str(step.status),
                }
            )

    pending_approvals = []
    approval_requests = state.get("approval_requests", [])
    for req in approval_requests:
        pending_approvals.append(
            {
                "approval_id": req.approval_id,
                "tool_name": req.tool_name,
                "arguments": req.arguments,
                "reason": req.reason,
                "permission_level": (
                    req.permission_level.value
                    if hasattr(req.permission_level, "value")
                    else str(req.permission_level)
                ),
            }
        )

    # Normalize status string to Component C requirements
    if status_str == WorkflowStatus.AWAITING_APPROVAL.value or pending_approvals:
        normalized_status = "waiting_for_approval"
        message = (
            f"Workflow {workflow_id} paused awaiting human approval for "
            f"protected operation '{pending_approvals[0]['tool_name']}'."
        )
    elif status_str == WorkflowStatus.COMPLETED.value:
        normalized_status = "completed"
        message = state.get("final_response") or "Workflow completed successfully."
    elif status_str == WorkflowStatus.FAILED.value:
        normalized_status = "failed"
        message = state.get("error_message") or "Workflow execution failed."
    elif status_str == WorkflowStatus.CANCELLED.value:
        normalized_status = "cancelled"
        message = "Workflow was cancelled."
    else:
        normalized_status = status_str

    return {
        "workflow_id": workflow_id,
        "thread_id": thread_id,
        "status": normalized_status,
        "message": message,
        "steps": steps_summary,
        "approval_required": len(pending_approvals) > 0,
        "pending_approvals": pending_approvals,
        "result": {
            "final_response": state.get("final_response"),
            "execution_results": {
                k: (v.model_dump() if hasattr(v, "model_dump") else v)
                for k, v in state.get("execution_results", {}).items()
            },
        },
    }


async def run_agent_request(
    request: str,
    thread_id: str | None = None,
    llm: Any | None = None,
    adapter: Any | None = None,
    registry: Any | None = None,
    silent: bool = False,
) -> dict[str, Any]:
    """Execute natural-language request through the LangGraph engine and render progress."""
    if not request or not request.strip():
        if not silent:
            render_error("User request cannot be empty.")
        return {
            "workflow_id": "",
            "thread_id": thread_id or "",
            "status": "failed",
            "message": "User request cannot be empty.",
            "steps": [],
            "approval_required": False,
            "pending_approvals": [],
            "result": {},
        }

    if not silent:
        console.print(f"\n[dim]Initializing workflow for request:[/dim] '{request.strip()}'...")

    try:
        state = await execute_agent_workflow(
            user_request=request.strip(),
            thread_id=thread_id,
            llm=llm,
            adapter=adapter,
            registry=registry,
        )
    except Exception as exc:
        logger.error("Workflow execution failed: %s", exc)
        if not silent:
            render_error(f"Execution failed: {exc}")
        return {
            "workflow_id": "",
            "thread_id": thread_id or "",
            "status": "failed",
            "message": str(exc),
            "steps": [],
            "approval_required": False,
            "pending_approvals": [],
            "result": {},
        }

    response_data = format_agent_response(state)

    if not silent:
        # Render execution progress
        plan = state.get("plan")
        if plan and hasattr(plan, "steps"):
            console.print("\n[bold]Execution Plan:[/bold]")
            for idx, step in enumerate(plan.steps, 1):
                render_progress_step(idx, step.description, step.tool_name)

        if response_data["approval_required"]:
            for appr in response_data["pending_approvals"]:
                render_approval_notice(
                    approval_id=appr["approval_id"],
                    tool_name=appr["tool_name"],
                    workflow_id=response_data["workflow_id"],
                    thread_id=response_data["thread_id"],
                    permission_level=appr["permission_level"],
                    arguments=appr["arguments"],
                    reason=appr["reason"],
                )
        elif response_data["status"] == "completed":
            final_text = response_data["result"].get("final_response") or response_data["message"]
            render_final_response(final_text)
        elif response_data["status"] == "failed":
            render_error(response_data["message"])

    return response_data
