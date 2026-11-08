"""Unit tests for the SYNQRA Tool Router Node and Dependency Resolution."""

import pytest

from synqra.agents.exceptions import SynqraAgentError
from synqra.agents.router import route_runnable_steps, router_node
from synqra.agents.schemas import ExecutionPlan, PermissionLevel, PlanStep, WorkflowStatus
from synqra.agents.state import AgentState
from synqra.agents.tools import build_agent_tool_registry


def test_router_parallel_independent_reads() -> None:
    """Verify router schedules independent READ steps together in parallel."""
    registry = build_agent_tool_registry()

    step1 = PlanStep(step_id="s1", description="Search web", tool_name="search_web")
    step2 = PlanStep(
        step_id="s2", description="Search github", tool_name="search_github_repositories"
    )
    step3 = PlanStep(
        step_id="s3",
        description="Create doc",
        tool_name="create_google_document",
        dependencies=["s1", "s2"],
        permission_level=PermissionLevel.WRITE,
    )
    plan = ExecutionPlan(user_request="Research and save", steps=[step1, step2, step3])

    # Initially with no completed steps: s1 and s2 are runnable
    batch = route_runnable_steps(
        plan=plan,
        completed_steps=[],
        pending_steps=["s1", "s2", "s3"],
        registry=registry,
        max_parallel=4,
    )
    assert batch is not None
    assert batch.step_ids == ["s1", "s2"]
    assert batch.is_parallel is True
    assert batch.permission_level == PermissionLevel.READ


def test_router_sequential_dependent_steps() -> None:
    """Verify router waits for prerequisites before scheduling dependent steps."""
    registry = build_agent_tool_registry()

    step1 = PlanStep(step_id="s1", description="Search web", tool_name="search_web")
    step2 = PlanStep(
        step_id="s2",
        description="Extract",
        tool_name="extract_webpage_content",
        dependencies=["s1"],
    )
    plan = ExecutionPlan(user_request="Search and extract", steps=[step1, step2])

    # S1 completed -> S2 is scheduled singularly
    batch = route_runnable_steps(
        plan=plan,
        completed_steps=["s1"],
        pending_steps=["s2"],
        registry=registry,
    )
    assert batch is not None
    assert batch.step_ids == ["s2"]
    assert batch.is_parallel is False


def test_router_single_write_isolation() -> None:
    """Verify WRITE operations are scheduled singularly and never in parallel."""
    registry = build_agent_tool_registry()

    step1 = PlanStep(
        step_id="w1",
        description="Upload file",
        tool_name="upload_drive_file",
        permission_level=PermissionLevel.WRITE,
    )
    step2 = PlanStep(
        step_id="w2",
        description="Create doc",
        tool_name="create_google_document",
        permission_level=PermissionLevel.WRITE,
    )
    plan = ExecutionPlan(user_request="Two writes", steps=[step1, step2])

    batch = route_runnable_steps(
        plan=plan,
        completed_steps=[],
        pending_steps=["w1", "w2"],
        registry=registry,
        max_parallel=4,
    )
    assert batch is not None
    assert batch.step_ids == ["w1"]
    assert batch.is_parallel is False
    assert batch.permission_level == PermissionLevel.WRITE


def test_router_rejects_unknown_tool() -> None:
    """Verify router raises SynqraAgentError if a step references an unregistered tool."""
    registry = build_agent_tool_registry()
    step1 = PlanStep(step_id="bad", description="Bad", tool_name="phantom_tool")
    plan = ExecutionPlan(user_request="Bad plan", steps=[step1])

    with pytest.raises(SynqraAgentError, match="references unknown tool"):
        route_runnable_steps(
            plan=plan,
            completed_steps=[],
            pending_steps=["bad"],
            registry=registry,
        )


@pytest.mark.asyncio
async def test_router_node_execution() -> None:
    """Verify router_node returns expected current_step_id."""
    registry = build_agent_tool_registry()
    step1 = PlanStep(step_id="s1", description="Search web", tool_name="search_web")
    plan = ExecutionPlan(user_request="Test", steps=[step1])

    state = AgentState(user_request="Test")
    state.set_execution_plan(plan)

    res = await router_node(state=state, registry=registry)
    assert res["current_step_id"] == "s1"
    assert res["workflow_status"] == WorkflowStatus.EXECUTING
