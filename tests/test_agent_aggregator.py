"""Unit tests for the SYNQRA Result Aggregator Node."""

import pytest

from synqra.agents.aggregator import (
    aggregator_node,
    extract_links_and_citations,
    format_deterministic_response,
)
from synqra.agents.schemas import ExecutionPlan, PlanStep, StepStatus, WorkflowStatus
from synqra.agents.state import AgentState


def test_extract_links_and_citations() -> None:
    """Verify recursive scanning for URLs and source citations."""
    results = {
        "step_1": {
            "results": [
                {"title": "Page 1", "url": "https://docs.python.org/3/"},
                {"title": "Repo", "html_url": "https://github.com/fastapi/fastapi"},
            ]
        },
        "step_2": {"watch_url": "https://youtube.com/watch?v=12345"},
    }
    links = extract_links_and_citations(results)
    assert "https://docs.python.org/3/" in links
    assert "https://github.com/fastapi/fastapi" in links
    assert "https://youtube.com/watch?v=12345" in links


def test_format_deterministic_response() -> None:
    """Verify deterministic formatting preserves step outcomes and links."""
    step = PlanStep(
        step_id="step_1",
        description="Search web",
        tool_name="search_web",
        status=StepStatus.COMPLETED,
    )
    plan = ExecutionPlan(user_request="Test summary", steps=[step])
    results = {"step_1": {"url": "https://fastapi.tiangolo.com/"}}

    summary = format_deterministic_response(
        user_request="Test summary",
        plan=plan,
        results=results,
        error_history=[],
        status=WorkflowStatus.COMPLETED,
    )

    assert "Test summary" in summary
    assert "https://fastapi.tiangolo.com/" in summary
    assert "COMPLETED" in summary


@pytest.mark.asyncio
async def test_aggregator_node_execution() -> None:
    """Verify aggregator_node sets final_response in AgentState."""
    step = PlanStep(
        step_id="step_1",
        description="Search web",
        tool_name="search_web",
        status=StepStatus.COMPLETED,
    )
    plan = ExecutionPlan(user_request="Find tools", steps=[step])

    state = AgentState(user_request="Find tools")
    state.set_execution_plan(plan)
    state.execution_results = {"step_1": {"url": "https://example.com"}}

    res = await aggregator_node(state=state)

    assert res["final_response"] is not None
    assert "Find tools" in res["final_response"]
    assert "https://example.com" in res["final_response"]
