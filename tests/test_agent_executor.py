"""Unit tests for the SYNQRA Executor Node."""

import pytest
from pydantic import BaseModel, Field

from synqra.agents.executor import executor_node
from synqra.agents.schemas import ExecutionPlan, PermissionLevel, PlanStep, WorkflowStatus
from synqra.agents.state import AgentState
from synqra.agents.tool_registry import AgentToolRegistry, RegisteredTool


class MockSearchInput(BaseModel):
    query: str = Field(description="Search term")


async def mock_search_fn(query: str) -> dict[str, str]:
    return {"result": f"found for {query}"}


async def mock_write_fn(query: str) -> dict[str, str]:
    return {"result": f"wrote {query}"}


@pytest.mark.asyncio
async def test_executor_node_successful_execution() -> None:
    """Verify executor_node executes scheduled READ steps successfully."""
    registry = AgentToolRegistry()
    registry.register_tool(
        RegisteredTool(
            name="mock_search",
            description="Search tool",
            integration_name="mock",
            input_schema=MockSearchInput,
            execution_function=mock_search_fn,
            permission_level=PermissionLevel.READ,
        )
    )

    step1 = PlanStep(
        step_id="step_1",
        description="Search task",
        tool_name="mock_search",
        arguments={"query": "agentic AI"},
    )
    plan = ExecutionPlan(user_request="Search AI", steps=[step1])

    state = AgentState(user_request="Search AI")
    state.set_execution_plan(plan)

    res = await executor_node(state=state, registry=registry)

    assert res["completed_steps"] == ["step_1"]
    assert res["pending_steps"] == []
    assert res["workflow_status"] == WorkflowStatus.COMPLETED
    assert "step_1" in res["execution_results"]
    assert res["execution_results"]["step_1"].success is True
    assert res["execution_results"]["step_1"].output == {"result": "found for agentic AI"}


@pytest.mark.asyncio
async def test_executor_node_blocks_unapproved_write() -> None:
    """Verify executor_node pauses and sets AWAITING_APPROVAL for unapproved WRITE steps."""
    registry = AgentToolRegistry()
    registry.register_tool(
        RegisteredTool(
            name="mock_write",
            description="Write tool",
            integration_name="mock",
            input_schema=MockSearchInput,
            execution_function=mock_write_fn,
            permission_level=PermissionLevel.WRITE,
        )
    )

    step1 = PlanStep(
        step_id="write_1",
        description="Write task",
        tool_name="mock_write",
        arguments={"query": "data"},
        permission_level=PermissionLevel.WRITE,
    )
    plan = ExecutionPlan(user_request="Write data", steps=[step1])

    state = AgentState(user_request="Write data")
    state.set_execution_plan(plan)

    res = await executor_node(state=state, registry=registry)

    assert res["workflow_status"] == WorkflowStatus.AWAITING_APPROVAL
    assert res["current_step_id"] == "write_1"
    assert len(res["approval_requests"]) == 1
    assert res["approval_requests"][0].tool_name == "mock_write"
