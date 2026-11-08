"""Unit tests for SYNQRA agent Pydantic v2 schemas and validation logic."""

from datetime import UTC, datetime, timedelta

import pytest

from synqra.agents.exceptions import ApprovalError, InvalidExecutionPlanError
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


def test_valid_plan_step_creation() -> None:
    """Verify a valid PlanStep instantiates with defaults and custom fields."""
    step = PlanStep(
        step_id="step_1",
        description="Search recent web developments",
        tool_name="search_web",
        arguments={"query": "agentic AI", "max_results": 3},
        permission_level=PermissionLevel.READ,
    )
    assert step.step_id == "step_1"
    assert step.tool_name == "search_web"
    assert step.status == StepStatus.PENDING
    assert step.dependencies == []
    assert step.permission_level == PermissionLevel.READ


def test_plan_step_self_dependency_rejected() -> None:
    """Verify a PlanStep cannot list itself as a dependency."""
    with pytest.raises(InvalidExecutionPlanError, match="cannot depend on itself"):
        PlanStep(
            step_id="step_1",
            description="Self-referential step",
            tool_name="search_web",
            dependencies=["step_1"],
        )


def test_plan_step_status_transitions() -> None:
    """Verify valid and invalid state transitions on a PlanStep."""
    step = PlanStep(
        step_id="step_1",
        description="Test status step",
        tool_name="search_web",
    )
    assert step.status == StepStatus.PENDING

    # PENDING -> RUNNING is valid
    step.transition_status(StepStatus.RUNNING)
    assert step.status == StepStatus.RUNNING

    # RUNNING -> COMPLETED is valid
    step.transition_status(StepStatus.COMPLETED)
    assert step.status == StepStatus.COMPLETED

    # COMPLETED -> RUNNING is invalid
    with pytest.raises(
        InvalidExecutionPlanError, match="cannot transition from completed to running"
    ):
        step.transition_status(StepStatus.RUNNING)


def test_valid_execution_plan_creation() -> None:
    """Verify a multi-step execution plan with valid dependency order."""
    step1 = PlanStep(
        step_id="search",
        description="Search web",
        tool_name="search_web",
        arguments={"query": "python"},
    )
    step2 = PlanStep(
        step_id="extract",
        description="Extract webpage content",
        tool_name="extract_webpage_content",
        dependencies=["search"],
    )
    step3 = PlanStep(
        step_id="calendar",
        description="Schedule study session",
        tool_name="create_calendar_event",
        dependencies=["extract"],
        permission_level=PermissionLevel.SENSITIVE,
    )

    plan = ExecutionPlan(
        user_request="Research python and schedule study session",
        steps=[step1, step2, step3],
        reasoning_summary="Linear 3-step workflow",
    )

    assert plan.estimated_steps == 3
    assert plan.status == WorkflowStatus.PENDING
    assert plan.get_execution_order() == ["search", "extract", "calendar"]
    assert plan.get_parallel_execution_tiers() == [["search"], ["extract"], ["calendar"]]


def test_execution_plan_duplicate_step_ids() -> None:
    """Verify that duplicate step IDs are strictly rejected."""
    step1 = PlanStep(step_id="duplicate", description="A", tool_name="tool_a")
    step2 = PlanStep(step_id="duplicate", description="B", tool_name="tool_b")

    with pytest.raises(InvalidExecutionPlanError, match="Duplicate step IDs found"):
        ExecutionPlan(
            user_request="Test request",
            steps=[step1, step2],
        )


def test_execution_plan_missing_dependency() -> None:
    """Verify that referencing an undefined step ID raises an error."""
    step1 = PlanStep(
        step_id="step_1",
        description="Step 1",
        tool_name="tool_a",
        dependencies=["non_existent_step"],
    )

    with pytest.raises(InvalidExecutionPlanError, match="references non-existent dependency"):
        ExecutionPlan(
            user_request="Test request",
            steps=[step1],
        )


def test_execution_plan_circular_dependency_detected() -> None:
    """Verify that circular dependencies (A -> B -> A) are detected and rejected."""
    step1 = PlanStep(step_id="step_a", description="A", tool_name="tool_a", dependencies=["step_b"])
    step2 = PlanStep(step_id="step_b", description="B", tool_name="tool_b", dependencies=["step_a"])

    with pytest.raises(InvalidExecutionPlanError, match="Circular dependency detected"):
        ExecutionPlan(
            user_request="Cyclic request",
            steps=[step1, step2],
        )


def test_execution_plan_parallel_tiers() -> None:
    """Verify parallel tier computation for independent steps."""
    # step_1 and step_2 are independent
    step1 = PlanStep(step_id="step_1", description="Search web", tool_name="search_web")
    step2 = PlanStep(step_id="step_2", description="Search GitHub", tool_name="search_github")
    # step_3 depends on both
    step3 = PlanStep(
        step_id="step_3",
        description="Aggregate and create doc",
        tool_name="create_google_document",
        dependencies=["step_1", "step_2"],
    )

    plan = ExecutionPlan(
        user_request="Research across web and github then save doc",
        steps=[step1, step2, step3],
    )

    tiers = plan.get_parallel_execution_tiers()
    assert len(tiers) == 2
    assert set(tiers[0]) == {"step_1", "step_2"}
    assert tiers[1] == ["step_3"]


def test_tool_execution_result_schema() -> None:
    """Verify ToolExecutionResult creation and immutability."""
    res = ToolExecutionResult(
        tool_name="search_web",
        success=True,
        output={"results": [{"title": "Agentic AI"}]},
        execution_duration=0.45,
    )
    assert res.tool_name == "search_web"
    assert res.success is True
    assert res.error is None
    assert res.execution_duration == 0.45
    assert res.execution_id.startswith("exec_")


def test_approval_request_validation() -> None:
    """Verify ApprovalRequest schema and expiration checking."""
    now = datetime.now(UTC)
    future = now + timedelta(minutes=15)
    past = now - timedelta(minutes=5)

    req = ApprovalRequest(
        workflow_id="wf_123",
        tool_name="send_gmail_message",
        arguments={"to": "user@example.com", "subject": "Test"},
        reason="Sending an email directly to external recipient",
        expires_at=future,
    )
    assert req.status == ApprovalStatus.PENDING
    assert req.is_expired(now) is False
    assert req.is_expired(now + timedelta(minutes=20)) is True

    # Ensure invalid expiration (in past relative to created_at) is rejected
    with pytest.raises(ApprovalError, match="must be after created_at"):
        ApprovalRequest(
            workflow_id="wf_123",
            tool_name="send_gmail_message",
            arguments={},
            reason="Test",
            created_at=now,
            expires_at=past,
        )


def test_approval_decision_validation() -> None:
    """Verify ApprovalDecision enforces human decision-makers and blocks automated agents."""
    valid_decision = ApprovalDecision(
        approval_id="appr_001",
        approved=True,
        decided_by="admin_user_alice",
        reason="Verified parameters are safe",
    )
    assert valid_decision.approved is True
    assert valid_decision.decided_by == "admin_user_alice"

    # Reject empty decided_by
    with pytest.raises(ApprovalError, match="non-empty decided_by"):
        ApprovalDecision(
            approval_id="appr_001",
            approved=True,
            decided_by="   ",
        )

    # Reject automated agent / LLM approval
    disallowed = ["llm", "ai", "agent", "assistant", "system", "gpt-4o", "claude", "bot"]
    for actor in disallowed:
        with pytest.raises(ApprovalError, match="The LLM must never generate or approve"):
            ApprovalDecision(
                approval_id="appr_001",
                approved=True,
                decided_by=actor,
            )
