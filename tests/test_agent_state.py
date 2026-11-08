"""Unit tests for SYNQRA AgentState lifecycle, step tracking, and resumption."""

import pytest

from synqra.agents.exceptions import InvalidAgentStateError
from synqra.agents.schemas import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStatus,
    ExecutionPlan,
    PermissionLevel,
    PlanStep,
    StepStatus,
    ToolExecutionResult,
    WorkflowStatus,
)
from synqra.agents.state import AgentState


def test_agent_state_initialization() -> None:
    """Verify initial blank AgentState properties."""
    state = AgentState(user_request="Research topic")
    assert state.user_request == "Research topic"
    assert state.workflow_id.startswith("wf_")
    assert state.thread_id.startswith("th_")
    assert state.workflow_status == WorkflowStatus.PENDING
    assert state.completed_steps == []
    assert state.pending_steps == []
    assert state.execution_plan is None
    assert state.current_step_id is None
    assert state.retry_count == 0
    assert state.is_resumable() is False


def test_agent_state_inconsistency_rejected() -> None:
    """Verify that a step cannot simultaneously be pending and completed."""
    with pytest.raises(InvalidAgentStateError, match="cannot be both completed and pending"):
        AgentState(
            user_request="Inconsistent",
            completed_steps=["step_1"],
            pending_steps=["step_1"],
        )


def test_agent_state_set_execution_plan() -> None:
    """Verify setting an execution plan populates pending steps in order."""
    step1 = PlanStep(step_id="step_a", description="A", tool_name="tool_a")
    step2 = PlanStep(step_id="step_b", description="B", tool_name="tool_b", dependencies=["step_a"])
    plan = ExecutionPlan(user_request="Two steps", steps=[step1, step2])

    state = AgentState(user_request="Two steps")
    state.set_execution_plan(plan)

    assert state.execution_plan is plan
    assert state.pending_steps == ["step_a", "step_b"]
    assert state.completed_steps == []
    assert state.workflow_status == WorkflowStatus.EXECUTING
    assert state.is_resumable() is True

    # Next runnable step is only step_a
    runnable = state.get_next_runnable_steps()
    assert len(runnable) == 1
    assert runnable[0].step_id == "step_a"


def test_agent_state_step_execution_lifecycle() -> None:
    """Verify stepping through execution, status transitions, and completion."""
    step1 = PlanStep(step_id="step_1", description="Step 1", tool_name="tool_a")
    step2 = PlanStep(
        step_id="step_2", description="Step 2", tool_name="tool_b", dependencies=["step_1"]
    )
    plan = ExecutionPlan(user_request="Workflow", steps=[step1, step2])

    state = AgentState(user_request="Workflow")
    state.set_execution_plan(plan)

    # 1. Mark step_1 running
    state.mark_step_running("step_1")
    assert state.current_step_id == "step_1"
    assert step1.status == StepStatus.RUNNING

    # 2. Mark step_1 completed
    res1 = ToolExecutionResult(tool_name="tool_a", success=True, output="output 1")
    state.mark_step_completed("step_1", res1)
    assert state.completed_steps == ["step_1"]
    assert state.pending_steps == ["step_2"]
    assert state.current_step_id is None
    assert step1.status == StepStatus.COMPLETED
    assert state.execution_results["step_1"] == res1

    # 3. Now step_2 becomes runnable
    runnable = state.get_next_runnable_steps()
    assert len(runnable) == 1
    assert runnable[0].step_id == "step_2"

    # 4. Mark step_2 running and completed
    state.mark_step_running("step_2")
    res2 = ToolExecutionResult(tool_name="tool_b", success=True, output="output 2")
    state.mark_step_completed("step_2", res2)

    assert state.completed_steps == ["step_1", "step_2"]
    assert state.pending_steps == []
    assert state.workflow_status == WorkflowStatus.COMPLETED


def test_agent_state_step_failure() -> None:
    """Verify step failure records error history and sets workflow status to FAILED."""
    step1 = PlanStep(step_id="step_1", description="Failing step", tool_name="bad_tool")
    plan = ExecutionPlan(user_request="Fail workflow", steps=[step1])

    state = AgentState(user_request="Fail workflow")
    state.set_execution_plan(plan)
    state.mark_step_running("step_1")

    state.mark_step_failed("step_1", "Network timeout contacting upstream service")
    assert step1.status == StepStatus.FAILED
    assert state.workflow_status == WorkflowStatus.FAILED
    assert len(state.error_history) == 1
    assert "Network timeout" in state.error_history[0]


def test_agent_state_human_approval_lifecycle() -> None:
    """Verify requesting approval, pausing workflow, and applying approval decisions."""
    step1 = PlanStep(
        step_id="send_email",
        description="Send email",
        tool_name="send_gmail_message",
        permission_level=PermissionLevel.SENSITIVE,
    )
    plan = ExecutionPlan(user_request="Email workflow", steps=[step1])

    state = AgentState(user_request="Email workflow")
    state.set_execution_plan(plan)

    appr_req = ApprovalRequest(
        workflow_id=state.workflow_id,
        tool_name="send_gmail_message",
        arguments={"to": "client@example.com"},
        reason="Requires sign-off before emailing external client",
    )
    state.request_approval(appr_req)

    assert state.workflow_status == WorkflowStatus.AWAITING_APPROVAL
    assert len(state.approval_requests) == 1
    assert state.is_resumable() is True

    # 1. Apply approval decision (Approved)
    decision = ApprovalDecision(
        approval_id=appr_req.approval_id,
        approved=True,
        decided_by="supervisor_bob",
    )
    state.apply_approval_decision(decision)

    assert appr_req.status == ApprovalStatus.APPROVED
    assert state.workflow_status == WorkflowStatus.EXECUTING

    # 2. Applying decision again raises error
    with pytest.raises(InvalidAgentStateError, match="already approved"):
        state.apply_approval_decision(decision)


def test_agent_state_approval_rejection() -> None:
    """Verify that rejecting approval sets status to CANCELLED and logs the reason."""
    step1 = PlanStep(
        step_id="delete_event", description="Delete event", tool_name="delete_calendar_event"
    )
    plan = ExecutionPlan(user_request="Delete event", steps=[step1])

    state = AgentState(user_request="Delete event")
    state.set_execution_plan(plan)

    appr_req = ApprovalRequest(
        workflow_id=state.workflow_id,
        tool_name="delete_calendar_event",
        arguments={"event_id": "evt_123"},
        reason="Confirm deletion",
    )
    state.request_approval(appr_req)

    rejection = ApprovalDecision(
        approval_id=appr_req.approval_id,
        approved=False,
        decided_by="supervisor_bob",
        reason="Event is still required",
    )
    state.apply_approval_decision(rejection)

    assert appr_req.status == ApprovalStatus.REJECTED
    assert state.workflow_status == WorkflowStatus.CANCELLED
    assert any("Event is still required" in err for err in state.error_history)


def test_agent_state_langgraph_dict_conversion() -> None:
    """Verify that AgentState exports to a LangGraph-compatible dictionary."""
    step1 = PlanStep(step_id="step_1", description="Step 1", tool_name="tool_a")
    plan = ExecutionPlan(user_request="Test export", steps=[step1])

    state = AgentState(user_request="Test export")
    state.set_execution_plan(plan)

    lg_dict = state.to_langgraph_dict()
    assert lg_dict["workflow_id"] == state.workflow_id
    assert lg_dict["user_request"] == "Test export"
    assert lg_dict["pending_steps"] == ["step_1"]
    assert lg_dict["workflow_status"] == WorkflowStatus.EXECUTING.value
    assert isinstance(lg_dict["created_at"], str)
