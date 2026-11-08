"""Unit tests for the SYNQRA Error Recovery Node."""

import pytest

from synqra.agents.recovery import recovery_node
from synqra.agents.schemas import (
    ExecutionPlan,
    PermissionLevel,
    PlanStep,
    StepStatus,
    WorkflowStatus,
)
from synqra.agents.state import AgentState


@pytest.mark.asyncio
async def test_recovery_node_retries_transient_read() -> None:
    """Verify recovery_node increments retry count and resets failed step for retry."""
    step = PlanStep(
        step_id="step_1",
        description="Search",
        tool_name="search_web",
        permission_level=PermissionLevel.READ,
        status=StepStatus.FAILED,
        error="Network timeout connecting to host",
    )
    plan = ExecutionPlan(user_request="Retry query", steps=[step])

    state = AgentState(user_request="Retry query")
    state.set_execution_plan(plan)
    state.retry_count = 0
    state.pending_steps = []

    res = await recovery_node(state=state, max_retries=3)

    assert res["workflow_status"] == WorkflowStatus.EXECUTING
    assert res["retry_count"] == 1
    assert res["pending_steps"] == ["step_1"]
    assert step.status == StepStatus.PENDING
    assert step.error is None


@pytest.mark.asyncio
async def test_recovery_node_blocks_write_retries() -> None:
    """Verify recovery_node refuses to retry failed WRITE operations."""
    step = PlanStep(
        step_id="write_1",
        description="Create doc",
        tool_name="create_google_document",
        permission_level=PermissionLevel.WRITE,
        status=StepStatus.FAILED,
        error="Network timeout",
    )
    plan = ExecutionPlan(user_request="No write retry", steps=[step])

    state = AgentState(user_request="No write retry")
    state.set_execution_plan(plan)

    res = await recovery_node(state=state)

    assert res["workflow_status"] == WorkflowStatus.FAILED
    assert any("cannot automatically retry WRITE" in err for err in res["error_history"])


@pytest.mark.asyncio
async def test_recovery_node_exhausts_retries() -> None:
    """Verify recovery_node marks failure when max retries are exceeded."""
    step = PlanStep(
        step_id="step_1",
        description="Search",
        tool_name="search_web",
        permission_level=PermissionLevel.READ,
        status=StepStatus.FAILED,
        error="Gateway timeout 504",
    )
    plan = ExecutionPlan(user_request="Max retry", steps=[step])

    state = AgentState(user_request="Max retry")
    state.set_execution_plan(plan)
    state.retry_count = 3  # Max reached

    res = await recovery_node(state=state, max_retries=3)

    assert res["workflow_status"] == WorkflowStatus.FAILED
    assert any("failed permanently after 3 retries" in err for err in res["error_history"])
