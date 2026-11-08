"""Error Recovery and Re-planning Node for SYNQRA LangGraph Agentic Engine.

Applies bounded retry policies exclusively to transient READ operations, prevents
unauthorized auto-retries on WRITE/SENSITIVE actions, and coordinates failure
diagnostics and replanning when tools fail or services are unavailable.
"""

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from synqra.agents.evaluator import (
    DEFAULT_MAX_RETRIES,
    is_error_transient,
)
from synqra.agents.schemas import ExecutionPlan, PermissionLevel, StepStatus, WorkflowStatus
from synqra.agents.state import AgentState

logger = logging.getLogger("synqra.agents.recovery")


async def recovery_node(
    state: AgentState | dict[str, Any],
    config: RunnableConfig | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict[str, Any]:
    """LangGraph node handling failure recovery, retry increments, and replanning."""
    _ = config
    if isinstance(state, dict):
        plan_raw = state.get("execution_plan")
        plan = ExecutionPlan.model_validate(plan_raw) if isinstance(plan_raw, dict) else plan_raw
        retry_count = int(state.get("retry_count", 0))
        pending = list(state.get("pending_steps", []))
        errors = list(state.get("error_history", []))
    else:
        plan = state.execution_plan
        retry_count = state.retry_count
        pending = list(state.pending_steps)
        errors = list(state.error_history)

    if not plan:
        return {
            "workflow_status": WorkflowStatus.FAILED,
            "error_history": errors + ["Recovery failed: no execution plan found."],
        }

    # Identify the failing step(s)
    failed_steps = [s for s in plan.steps if s.status == StepStatus.FAILED]
    if not failed_steps:
        # No failing steps: nothing to recover, resume execution
        return {"workflow_status": WorkflowStatus.EXECUTING}

    failed_step = failed_steps[0]

    # Rule 1: Never automatically retry WRITE or SENSITIVE operations
    if failed_step.permission_level in (PermissionLevel.WRITE, PermissionLevel.SENSITIVE):
        perm_str = failed_step.permission_level.value.upper()
        msg = (
            f"Recovery disallowed: cannot automatically retry {perm_str} "
            f"tool '{failed_step.tool_name}' on step '{failed_step.step_id}'."
        )
        logger.warning(msg)
        return {
            "workflow_status": WorkflowStatus.FAILED,
            "error_history": errors + [msg],
        }

    # Rule 2: Check retry budget and error type
    if is_error_transient(failed_step.error) and retry_count < max_retries:
        new_retry_count = retry_count + 1
        msg = (
            f"Scheduling retry {new_retry_count}/{max_retries} for step '{failed_step.step_id}' "
            f"(transient error: {failed_step.error})."
        )
        logger.info(msg)

        # Reset step status for retry
        failed_step.status = StepStatus.PENDING
        failed_step.error = None
        if failed_step.step_id not in pending:
            pending.insert(0, failed_step.step_id)

        return {
            "execution_plan": plan,
            "retry_count": new_retry_count,
            "pending_steps": pending,
            "workflow_status": WorkflowStatus.EXECUTING,
            "error_history": errors + [msg],
        }

    # Rule 3: Retries exhausted or non-transient error: terminate with failure
    exhausted_msg = (
        f"Step '{failed_step.step_id}' failed permanently after {retry_count} retries. "
        f"Error: {failed_step.error}"
    )
    logger.error(exhausted_msg)
    return {
        "workflow_status": WorkflowStatus.FAILED,
        "error_history": errors + [exhausted_msg],
    }
