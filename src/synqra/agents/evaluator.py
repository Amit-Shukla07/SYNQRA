"""Evaluator Node for SYNQRA LangGraph Agentic Engine.

Inspects tool execution outcomes, checks whether prerequisite steps have yielded
sufficient results, determines whether failures are safely retryable (transient READ only),
and routes the workflow to completion, continuation, recovery, or approval pauses.
"""

import logging
from enum import StrEnum
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from synqra.agents.schemas import ExecutionPlan, PermissionLevel, StepStatus, WorkflowStatus
from synqra.agents.state import AgentState

logger = logging.getLogger("synqra.agents.evaluator")

DEFAULT_MAX_RETRIES: int = 3

TRANSIENT_ERROR_MARKERS: frozenset[str] = frozenset(
    {
        "timeout",
        "timed out",
        "connection",
        "temporarily unavailable",
        "rate limit",
        "429",
        "503",
        "504",
        "network",
    }
)


class EvaluatorAction(StrEnum):
    """Routing action decided by the evaluator node."""

    CONTINUE = "continue"
    COMPLETE = "complete"
    RETRY = "retry"
    FAIL = "fail"
    AWAITING_APPROVAL = "awaiting_approval"


class EvaluationResult(BaseModel):
    """Structured decision output from the evaluator."""

    action: EvaluatorAction = Field(description="Next workflow routing action.")
    reason: str = Field(description="Explanation of the evaluation decision.")
    retryable_step_id: str | None = Field(
        default=None,
        description="ID of the failed step if eligible for bounded retry.",
    )


def is_error_transient(error_message: str | None) -> bool:
    """Check whether an error message indicates a transient, retryable failure."""
    if not error_message:
        return False
    lower_err = error_message.lower()
    return any(marker in lower_err for marker in TRANSIENT_ERROR_MARKERS)


def evaluate_workflow_state(
    state: AgentState | dict[str, Any],
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> EvaluationResult:
    """Evaluate current workflow progress and determine next action without LLM overhead."""
    # 1. Check if workflow is awaiting human approval
    if isinstance(state, dict):
        status = state.get("workflow_status", WorkflowStatus.EXECUTING)
        plan_raw = state.get("execution_plan")
        plan = ExecutionPlan.model_validate(plan_raw) if isinstance(plan_raw, dict) else plan_raw
        pending = list(state.get("pending_steps", []))
        retry_count = int(state.get("retry_count", 0))
    else:
        status = state.workflow_status
        plan = state.execution_plan
        pending = list(state.pending_steps)
        retry_count = state.retry_count

    if status == WorkflowStatus.AWAITING_APPROVAL:
        return EvaluationResult(
            action=EvaluatorAction.AWAITING_APPROVAL,
            reason="Workflow is paused awaiting human approval for sensitive action.",
        )

    if not plan:
        return EvaluationResult(
            action=EvaluatorAction.FAIL,
            reason="Workflow has no execution plan.",
        )

    # 1b. Check for explicit rejection
    if status in (WorkflowStatus.REJECTED, WorkflowStatus.CANCELLED) or any(
        s.status == StepStatus.REJECTED for s in plan.steps
    ):
        return EvaluationResult(
            action=EvaluatorAction.FAIL,
            reason="Workflow or step was rejected by human operator.",
        )

    # 2. Check for step failures
    failed_steps = [s for s in plan.steps if s.status == StepStatus.FAILED]
    if failed_steps:
        failed_step = failed_steps[0]
        # Never automatically retry WRITE or SENSITIVE operations
        if failed_step.permission_level != PermissionLevel.READ:
            return EvaluationResult(
                action=EvaluatorAction.FAIL,
                reason=(
                    f"Step '{failed_step.step_id}' ({failed_step.permission_level.value}) failed. "
                    f"Automatic retries are disallowed for write/sensitive operations."
                ),
            )

        # Check if error is transient and retry count allows
        if is_error_transient(failed_step.error) and retry_count < max_retries:
            return EvaluationResult(
                action=EvaluatorAction.RETRY,
                reason=(
                    f"Step '{failed_step.step_id}' transient error: {failed_step.error}. "
                    f"Attempting retry {retry_count + 1}/{max_retries}."
                ),
                retryable_step_id=failed_step.step_id,
            )

        return EvaluationResult(
            action=EvaluatorAction.FAIL,
            reason=(
                f"Step '{failed_step.step_id}' failed unrecoverably or exceeded retry limit: "
                f"{failed_step.error}."
            ),
        )

    # 3. Check completion
    if not pending:
        return EvaluationResult(
            action=EvaluatorAction.COMPLETE,
            reason="All planned execution steps completed successfully.",
        )

    # 4. More steps remain to be executed
    return EvaluationResult(
        action=EvaluatorAction.CONTINUE,
        reason=f"{len(pending)} step(s) remaining in execution queue.",
    )


async def evaluator_node(
    state: AgentState | dict[str, Any],
    config: RunnableConfig | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict[str, Any]:
    """LangGraph node assessing execution progress and routing next graph edge."""
    _ = config
    eval_result = evaluate_workflow_state(state=state, max_retries=max_retries)
    logger.debug("Evaluator decision: %s (%s)", eval_result.action.value, eval_result.reason)

    if eval_result.action == EvaluatorAction.AWAITING_APPROVAL:
        return {"workflow_status": WorkflowStatus.AWAITING_APPROVAL}

    if eval_result.action == EvaluatorAction.COMPLETE:
        return {"workflow_status": WorkflowStatus.COMPLETED}

    if eval_result.action == EvaluatorAction.FAIL:
        curr_status = (
            state.get("workflow_status") if isinstance(state, dict) else state.workflow_status
        )
        final_fail = (
            curr_status
            if curr_status in (WorkflowStatus.REJECTED, WorkflowStatus.CANCELLED)
            else WorkflowStatus.FAILED
        )
        return {"workflow_status": final_fail}

    return {}
