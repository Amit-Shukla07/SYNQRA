"""Unit tests for the SYNQRA Planner Node."""

import pytest

from synqra.agents.exceptions import InvalidExecutionPlanError
from synqra.agents.llm import get_mock_llm
from synqra.agents.planner import (
    PlannerOutput,
    PlanStepProposal,
    format_tools_catalog,
    plan_workflow,
    planner_node,
)
from synqra.agents.schemas import PermissionLevel, WorkflowStatus
from synqra.agents.state import AgentState
from synqra.agents.tools import build_agent_tool_registry


def test_format_tools_catalog() -> None:
    """Verify tool formatting for the planner system prompt."""
    registry = build_agent_tool_registry()
    catalog_str = format_tools_catalog(registry.list_tools())
    assert "search_youtube_videos" in catalog_str
    assert "search_web" in catalog_str
    assert "search_github_repositories" in catalog_str


@pytest.mark.asyncio
async def test_plan_workflow_simple_request() -> None:
    """Verify planning a single-step request with mock LLM structured output."""
    mock_output = PlannerOutput(
        reasoning_summary="Simple web search request",
        steps=[
            PlanStepProposal(
                step_id="step_1",
                description="Search for python 3.12 highlights",
                tool_name="search_web",
                arguments={"query": "python 3.12 highlights", "max_results": 3},
                dependencies=[],
                permission_level=PermissionLevel.READ,
            )
        ],
    )
    mock_llm = get_mock_llm(structured_response=mock_output)

    plan = await plan_workflow(
        user_request="Find python 3.12 highlights",
        llm=mock_llm,
    )

    assert plan.user_request == "Find python 3.12 highlights"
    assert len(plan.steps) == 1
    assert plan.steps[0].step_id == "step_1"
    assert plan.steps[0].tool_name == "search_web"
    assert plan.steps[0].permission_level == PermissionLevel.READ


@pytest.mark.asyncio
async def test_plan_workflow_multi_step_request() -> None:
    """Verify planning a multi-step request with dependencies."""
    mock_output = PlannerOutput(
        reasoning_summary="Two-step research and doc creation workflow",
        steps=[
            PlanStepProposal(
                step_id="search",
                description="Search web for agentic AI",
                tool_name="search_web",
                arguments={"query": "agentic AI"},
                dependencies=[],
            ),
            PlanStepProposal(
                step_id="save_doc",
                description="Create doc with findings",
                tool_name="create_google_document",
                arguments={"title": "Agentic AI Notes", "content": "Findings"},
                dependencies=["search"],
                permission_level=PermissionLevel.WRITE,
            ),
        ],
    )
    mock_llm = get_mock_llm(structured_response=mock_output)

    plan = await plan_workflow(
        user_request="Research agentic AI and save to Google Doc",
        llm=mock_llm,
    )

    assert len(plan.steps) == 2
    assert plan.get_execution_order() == ["search", "save_doc"]
    # Verify canonical permission enforcement: create_google_document must be WRITE
    assert plan.steps[1].permission_level == PermissionLevel.WRITE


@pytest.mark.asyncio
async def test_plan_workflow_rejects_unknown_tool() -> None:
    """Verify planner rejects proposed tools not in the registry."""
    mock_output = PlannerOutput(
        reasoning_summary="Invalid tool",
        steps=[
            PlanStepProposal(
                step_id="step_1",
                description="Invalid action",
                tool_name="non_existent_hack_tool",
                arguments={},
            )
        ],
    )
    mock_llm = get_mock_llm(structured_response=mock_output)

    with pytest.raises(InvalidExecutionPlanError, match="unknown or disabled tool"):
        await plan_workflow(
            user_request="Do invalid action",
            llm=mock_llm,
        )


@pytest.mark.asyncio
async def test_plan_workflow_max_steps_exceeded() -> None:
    """Verify planner enforces the configured max_steps constraint."""
    steps = [
        PlanStepProposal(
            step_id=f"step_{i}",
            description=f"Step {i}",
            tool_name="search_web",
            arguments={"query": f"term_{i}"},
        )
        for i in range(5)
    ]
    mock_output = PlannerOutput(reasoning_summary="Too many steps", steps=steps)
    mock_llm = get_mock_llm(structured_response=mock_output)

    with pytest.raises(InvalidExecutionPlanError, match="exceeds maximum step limit"):
        await plan_workflow(
            user_request="Big request",
            llm=mock_llm,
            max_steps=3,
        )


@pytest.mark.asyncio
async def test_planner_node_execution() -> None:
    """Verify planner_node returns expected AgentState dictionary update."""
    mock_output = PlannerOutput(
        reasoning_summary="Simple search",
        steps=[
            PlanStepProposal(
                step_id="step_1",
                description="Search web",
                tool_name="search_web",
                arguments={"query": "test"},
            )
        ],
    )
    mock_llm = get_mock_llm(structured_response=mock_output)
    state = AgentState(user_request="Search for test")

    res = await planner_node(state=state, llm=mock_llm)
    assert res["workflow_status"] == WorkflowStatus.EXECUTING
    assert res["pending_steps"] == ["step_1"]
    assert res["completed_steps"] == []
    assert res["execution_plan"] is not None
