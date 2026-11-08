"""Tests for the synqra_run_agent MCP tool (Phase 9.6 Component B)."""

from __future__ import annotations

import pytest

from synqra.agents.llm import MockChatModel
from synqra.agents.planner import PlannerOutput, PlanStepProposal
from synqra.agents.tool_adapter import AgentToolAdapter
from synqra.server.mcp_server import create_mcp_server


class DummyAdapter(AgentToolAdapter):
    """Mock tool adapter for MCP agent tool tests."""

    async def execute(self, tool=None, arguments=None, timeout=None, **_kwargs):
        _ = (tool, arguments, timeout)
        return {"results": ["FastAPI documentation content."]}


@pytest.mark.asyncio
async def test_mcp_agent_tool_registration() -> None:
    """Verify synqra_run_agent tool is properly registered on FastMCP server."""
    server = create_mcp_server()
    tools = await server.list_tools()
    tool_names = [t.name for t in tools]

    assert "synqra_run_agent" in tool_names
    agent_tool = next(t for t in tools if t.name == "synqra_run_agent")
    assert agent_tool.description is not None
    assert "orchestration engine" in agent_tool.description.lower()

    schema = (
        agent_tool.parameters
        if hasattr(agent_tool, "parameters")
        else getattr(agent_tool, "inputSchema", {})
    )
    assert "request" in schema.get("properties", {})
    assert "thread_id" in schema.get("properties", {})
    assert "request" in schema.get("required", [])
    assert "thread_id" not in schema.get("required", [])


@pytest.mark.asyncio
async def test_mcp_agent_tool_invocation_read_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify invoking synqra_run_agent tool with read-only request."""
    llm = MockChatModel(
        structured_response=PlannerOutput(
            reasoning_summary="Search web for FastAPI documentation",
            steps=[
                PlanStepProposal(
                    step_id="step_1",
                    tool_name="search_web",
                    arguments={"query": "FastAPI documentation"},
                    description="Search web",
                )
            ],
        ),
    )
    # Patch default get_llm to return MockChatModel and executor to use DummyAdapter
    monkeypatch.setattr("synqra.agents.graph.get_llm", lambda: llm)
    monkeypatch.setattr("synqra.agents.tools.get_default_tool_adapter", lambda: DummyAdapter())

    server = create_mcp_server()
    result = await server.call_tool(
        "synqra_run_agent", {"request": "Search web for FastAPI documentation"}
    )

    assert result.is_error is False
    assert result.structured_content is not None
    data = result.structured_content

    assert "workflow_id" in data
    assert "thread_id" in data
    assert "status" in data
    assert data["status"] in ("completed", "planning")
    assert data["approval_required"] is False
    assert len(data["pending_approvals"]) == 0


@pytest.mark.asyncio
async def test_mcp_agent_tool_invocation_approval_interruption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify invoking synqra_run_agent with write request halts on approval."""
    llm = MockChatModel(
        structured_response=PlannerOutput(
            reasoning_summary="Send notification email",
            steps=[
                PlanStepProposal(
                    step_id="step_1",
                    tool_name="send_gmail_message",
                    arguments={
                        "recipient": "boss@example.com",
                        "subject": "Alert",
                        "body": "Report",
                    },
                    description="Send email",
                )
            ],
        ),
    )
    monkeypatch.setattr("synqra.agents.graph.get_llm", lambda: llm)

    server = create_mcp_server()
    result = await server.call_tool(
        "synqra_run_agent", {"request": "Send email to boss@example.com"}
    )

    assert result.is_error is False
    data = result.structured_content

    assert data["status"] == "waiting_for_approval"
    assert data["approval_required"] is True
    assert len(data["pending_approvals"]) == 1
    assert data["pending_approvals"][0]["tool_name"] == "send_gmail_message"
    assert data["pending_approvals"][0]["arguments"]["recipient"] == "boss@example.com"


@pytest.mark.asyncio
async def test_mcp_agent_tool_empty_request_rejected() -> None:
    """Verify synqra_run_agent with empty request returns failure response."""
    server = create_mcp_server()
    result = await server.call_tool("synqra_run_agent", {"request": "   "})

    assert result.is_error is False
    data = result.structured_content
    assert data["status"] == "failed"
    assert "cannot be empty" in data["message"].lower()


@pytest.mark.asyncio
async def test_mcp_agent_tool_thread_id_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify synqra_run_agent correctly handles both omitted and explicit thread_id."""
    llm = MockChatModel(
        structured_response=PlannerOutput(
            reasoning_summary="Search web",
            steps=[
                PlanStepProposal(
                    step_id="step_1",
                    tool_name="search_web",
                    arguments={"query": "test"},
                    description="Search web",
                )
            ],
        ),
    )
    monkeypatch.setattr("synqra.agents.graph.get_llm", lambda: llm)
    monkeypatch.setattr("synqra.agents.tools.get_default_tool_adapter", lambda: DummyAdapter())

    server = create_mcp_server()

    # Case 1: Omitted thread_id -> server allocates a new thread_id
    res_omitted = await server.call_tool("synqra_run_agent", {"request": "test query 1"})
    assert res_omitted.is_error is False
    data_omitted = res_omitted.structured_content
    assert data_omitted["thread_id"].startswith("th_")

    # Case 2: Explicitly supplied thread_id -> server preserves and echoes the exact thread_id
    custom_thread = "th_custom_session_12345"
    res_explicit = await server.call_tool(
        "synqra_run_agent",
        {"request": "test query 2", "thread_id": custom_thread},
    )
    assert res_explicit.is_error is False
    data_explicit = res_explicit.structured_content
    assert data_explicit["thread_id"] == custom_thread
