"""End-to-end integration tests for the SYNQRA LangGraph Agentic Engine."""

import pytest
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from synqra.agents.graph import create_agent_graph, run_agent_workflow
from synqra.agents.llm import get_mock_llm
from synqra.agents.planner import PlannerOutput, PlanStepProposal
from synqra.agents.schemas import PermissionLevel, WorkflowStatus
from synqra.agents.tool_registry import AgentToolRegistry, RegisteredTool


class MockSearchSchema(BaseModel):
    query: str = Field(description="Search term")


async def mock_search_coro(query: str) -> dict[str, str]:
    return {"status": "ok", "url": "https://docs.python.org/3/", "query": query}


async def mock_write_coro(query: str) -> dict[str, str]:
    return {"status": "created", "doc_id": "doc_12345", "query": query}


@pytest.mark.asyncio
async def test_graph_simple_request_end_to_end() -> None:
    """Verify executing a simple single-step workflow through the full LangGraph pipeline."""
    # 1. Mock tool
    registry = AgentToolRegistry()
    registry.register_tool(
        RegisteredTool(
            name="search_web",
            description="Web search tool",
            integration_name="mock",
            input_schema=MockSearchSchema,
            execution_function=mock_search_coro,
            permission_level=PermissionLevel.READ,
        )
    )

    # 2. Mock planner output
    mock_plan = PlannerOutput(
        reasoning_summary="Single search plan",
        steps=[
            PlanStepProposal(
                step_id="step_1",
                description="Search for python documentation",
                tool_name="search_web",
                arguments={"query": "python"},
            )
        ],
    )
    mock_llm = get_mock_llm(structured_response=mock_plan)

    # 3. Create graph with in-memory checkpointer
    saver = MemorySaver()
    graph = create_agent_graph(llm=mock_llm, registry=registry, checkpointer=saver)

    # 4. Execute workflow
    result = await run_agent_workflow(
        user_request="Find python documentation",
        thread_id="test_thread_1",
        graph=graph,
    )

    assert result["workflow_status"] == WorkflowStatus.COMPLETED
    assert result["completed_steps"] == ["step_1"]
    assert "https://docs.python.org/3/" in result["final_response"]

    # 5. Verify checkpoint saved in MemorySaver
    saved_state = saver.get({"configurable": {"thread_id": "test_thread_1"}})
    assert saved_state is not None


@pytest.mark.asyncio
async def test_graph_approval_required_pause() -> None:
    """Verify that encountering an unapproved WRITE step halts the graph in AWAITING_APPROVAL."""
    registry = AgentToolRegistry()
    registry.register_tool(
        RegisteredTool(
            name="upload_drive_file",
            description="Upload file",
            integration_name="mock",
            input_schema=MockSearchSchema,
            execution_function=mock_write_coro,
            permission_level=PermissionLevel.WRITE,
        )
    )

    mock_plan = PlannerOutput(
        reasoning_summary="Write plan requiring approval",
        steps=[
            PlanStepProposal(
                step_id="write_step",
                description="Upload file to drive",
                tool_name="upload_drive_file",
                arguments={"query": "my_file.txt"},
                permission_level=PermissionLevel.WRITE,
            )
        ],
    )
    mock_llm = get_mock_llm(structured_response=mock_plan)

    graph = create_agent_graph(llm=mock_llm, registry=registry)
    result = await run_agent_workflow(
        user_request="Upload file to Drive",
        thread_id="test_thread_approval",
        graph=graph,
    )

    # Graph must fail closed and pause without executing the tool
    assert result["workflow_status"] == WorkflowStatus.AWAITING_APPROVAL
    assert result["current_step_id"] == "write_step"
    assert len(result["approval_requests"]) == 1
    assert result["approval_requests"][0].tool_name == "upload_drive_file"
