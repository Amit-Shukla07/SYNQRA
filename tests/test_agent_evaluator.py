"""Unit tests for the SYNQRA Evaluator Node."""

import pytest

from synqra.agents.evaluator import (
    EvaluatorAction,
    evaluate_workflow_state,
    evaluator_node,
    is_error_transient,
)
from synqra.agents.schemas import (
    ExecutionPlan,
    PermissionLevel,
    PlanStep,
    StepStatus,
    WorkflowStatus,
)
from synqra.agents.state import AgentState


def test_is_error_transient_markers() -> None:
    """Verify detection of transient errors vs deterministic errors."""
    assert is_error_transient("HTTP 504 Gateway Timeout") is True
    assert is_error_transient("Connection reset by peer") is True
    assert is_error_transient("Rate limit exceeded (429)") is True
    assert is_error_transient("Invalid arguments: missing 'query'") is False
    assert is_error_transient(None) is False


def test_evaluator_complete_when_all_steps_done() -> None:
    """Verify Evaluator signals COMPLETE when pending_steps is empty and no errors exist."""
    step = PlanStep(step_id="s1", description="A", tool_name="tool_a", status=StepStatus.COMPLETED)
    plan = ExecutionPlan(user_request="Done", steps=[step])

    state = AgentState(user_request="Done")
    state.set_execution_plan(plan)
    state.completed_steps = ["s1"]
    state.pending_steps = []

    decision = evaluate_workflow_state(state)
    assert decision.action == EvaluatorAction.COMPLETE


def test_evaluator_continue_when_steps_remain() -> None:
    """Verify Evaluator signals CONTINUE when pending_steps remain."""
    step1 = PlanStep(step_id="s1", description="A", tool_name="tool_a", status=StepStatus.COMPLETED)
    step2 = PlanStep(step_id="s2", description="B", tool_name="tool_b", status=StepStatus.PENDING)
    plan = ExecutionPlan(user_request="Pending", steps=[step1, step2])

    state = AgentState(user_request="Pending")
    state.set_execution_plan(plan)
    state.completed_steps = ["s1"]
    state.pending_steps = ["s2"]

    decision = evaluate_workflow_state(state)
    assert decision.action == EvaluatorAction.CONTINUE


def test_evaluator_retry_for_transient_read_failure() -> None:
    """Verify Evaluator signals RETRY for transient errors on READ operations."""
    step = PlanStep(
        step_id="s1",
        description="A",
        tool_name="search_web",
        permission_level=PermissionLevel.READ,
        status=StepStatus.FAILED,
        error="Connection timeout after 60 seconds",
    )
    plan = ExecutionPlan(user_request="Retry test", steps=[step])

    state = AgentState(user_request="Retry test")
    state.set_execution_plan(plan)
    state.retry_count = 0

    decision = evaluate_workflow_state(state, max_retries=3)
    assert decision.action == EvaluatorAction.RETRY
    assert decision.retryable_step_id == "s1"


def test_evaluator_fails_for_write_step_failure() -> None:
    """Verify Evaluator never retries WRITE failures automatically."""
    step = PlanStep(
        step_id="w1",
        description="Upload",
        tool_name="upload_drive_file",
        permission_level=PermissionLevel.WRITE,
        status=StepStatus.FAILED,
        error="Connection timeout",
    )
    plan = ExecutionPlan(user_request="Write fail", steps=[step])

    state = AgentState(user_request="Write fail")
    state.set_execution_plan(plan)

    decision = evaluate_workflow_state(state)
    assert decision.action == EvaluatorAction.FAIL
    assert "Automatic retries are disallowed for write/sensitive" in decision.reason


@pytest.mark.asyncio
async def test_evaluator_node_awaiting_approval() -> None:
    """Verify evaluator_node updates state properly when in AWAITING_APPROVAL."""
    state = AgentState(user_request="Approval wait")
    state.workflow_status = WorkflowStatus.AWAITING_APPROVAL

    res = await evaluator_node(state)
    assert res == {"workflow_status": WorkflowStatus.AWAITING_APPROVAL}
