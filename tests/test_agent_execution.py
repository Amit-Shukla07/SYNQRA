"""Unit tests for the centralized agent tool execution engine."""

from typing import Any

import pytest
from pydantic import BaseModel, Field

from synqra.agents.permissions import ApprovalContext
from synqra.agents.schemas import (
    ApprovalDecision,
    ApprovalRequest,
    PermissionLevel,
    ToolExecutionResult,
)
from synqra.agents.tool_registry import AgentToolRegistry, RegisteredTool
from synqra.agents.tools import execute_tool


class MockSearchInput(BaseModel):
    query: str = Field(description="Search term", min_length=1)
    limit: int = Field(default=5, ge=1, le=20)


class MockActionInput(BaseModel):
    message: str = Field(description="Action message")
    confirm: bool = Field(default=False)


async def mock_search_fn(query: str, limit: int = 5) -> dict[str, Any]:
    return {"query": query, "count": limit, "items": [f"item_{i}" for i in range(limit)]}


async def mock_action_fn(message: str, confirm: bool = False) -> dict[str, str]:
    return {"status": "sent", "message": message, "confirmed": str(confirm)}


@pytest.mark.asyncio
async def test_execute_tool_successful_read_only() -> None:
    """Verify executing a valid READ tool without approval."""
    registry = AgentToolRegistry()
    registry.register_tool(
        RegisteredTool(
            name="mock_search",
            description="Mock search",
            integration_name="mock_svc",
            input_schema=MockSearchInput,
            execution_function=mock_search_fn,
            permission_level=PermissionLevel.READ,
        )
    )

    res = await execute_tool(
        tool_name="mock_search",
        arguments={"query": "agentic", "limit": 2},
        workflow_id="wf_123",
        registry=registry,
    )

    assert isinstance(res, ToolExecutionResult)
    assert res.success is True
    assert res.error is None
    assert res.tool_name == "mock_search"
    assert res.output == {"query": "agentic", "count": 2, "items": ["item_0", "item_1"]}
    assert res.execution_duration is not None
    assert res.execution_duration >= 0.0


@pytest.mark.asyncio
async def test_execute_tool_invalid_arguments() -> None:
    """Verify passing invalid arguments returns ToolExecutionResult with success=False."""
    registry = AgentToolRegistry()
    registry.register_tool(
        RegisteredTool(
            name="mock_search",
            description="Mock search",
            integration_name="mock_svc",
            input_schema=MockSearchInput,
            execution_function=mock_search_fn,
            permission_level=PermissionLevel.READ,
        )
    )

    # query is empty (violates min_length=1) and limit is negative
    res = await execute_tool(
        tool_name="mock_search",
        arguments={"query": "", "limit": -5},
        workflow_id="wf_123",
        registry=registry,
    )

    assert res.success is False
    assert "Invalid arguments for tool 'mock_search'" in (res.error or "")
    assert res.output is None


@pytest.mark.asyncio
async def test_execute_tool_unknown_tool() -> None:
    """Verify executing an unregistered tool returns an error result."""
    registry = AgentToolRegistry()
    res = await execute_tool(
        tool_name="unregistered_tool",
        arguments={},
        workflow_id="wf_123",
        registry=registry,
    )

    assert res.success is False
    assert "not registered" in (res.error or "")


@pytest.mark.asyncio
async def test_execute_tool_disabled_tool() -> None:
    """Verify executing a disabled tool is blocked."""
    registry = AgentToolRegistry()
    registry.register_tool(
        RegisteredTool(
            name="mock_search",
            description="Mock search",
            integration_name="mock_svc",
            input_schema=MockSearchInput,
            execution_function=mock_search_fn,
            permission_level=PermissionLevel.READ,
            is_active=False,
        )
    )

    res = await execute_tool(
        tool_name="mock_search",
        arguments={"query": "test"},
        workflow_id="wf_123",
        registry=registry,
    )

    assert res.success is False
    assert "currently disabled" in (res.error or "")


@pytest.mark.asyncio
async def test_execute_tool_write_requires_approval() -> None:
    """Verify WRITE tool fails without approval and succeeds with approval."""
    registry = AgentToolRegistry()
    registry.register_tool(
        RegisteredTool(
            name="mock_action",
            description="Mock action",
            integration_name="mock_svc",
            input_schema=MockActionInput,
            execution_function=mock_action_fn,
            permission_level=PermissionLevel.WRITE,
        )
    )

    # 1. Without approval -> blocked
    res_no_appr = await execute_tool(
        tool_name="mock_action",
        arguments={"message": "hello", "confirm": False},
        workflow_id="wf_123",
        registry=registry,
    )
    assert res_no_appr.success is False
    assert "requires human approval" in (res_no_appr.error or "")

    # 2. With valid approval -> executed
    req = ApprovalRequest(
        workflow_id="wf_123",
        tool_name="mock_action",
        arguments={"message": "hello", "confirm": False},
        reason="Testing execution",
    )
    decision = ApprovalDecision(
        approval_id=req.approval_id, approved=True, decided_by="admin_alice"
    )
    ctx = ApprovalContext(approval_request=req, approval_decision=decision)

    res_appr = await execute_tool(
        tool_name="mock_action",
        arguments={"message": "hello", "confirm": False},
        workflow_id="wf_123",
        approval_context=ctx,
        registry=registry,
    )
    assert res_appr.success is True
    assert res_appr.output["status"] == "sent"


@pytest.mark.asyncio
async def test_execute_tool_failure_isolation() -> None:
    """Verify that an exception in a tool function does not crash the engine."""

    async def buggy_fn(query: str, limit: int = 5) -> str:
        _ = (query, limit)
        raise RuntimeError("Unexpected internal crash")

    registry = AgentToolRegistry()
    registry.register_tool(
        RegisteredTool(
            name="buggy_tool",
            description="Buggy tool",
            integration_name="mock_svc",
            input_schema=MockSearchInput,
            execution_function=buggy_fn,
            permission_level=PermissionLevel.READ,
        )
    )

    res = await execute_tool(
        tool_name="buggy_tool",
        arguments={"query": "test"},
        workflow_id="wf_123",
        registry=registry,
    )

    assert res.success is False
    assert "Unexpected internal crash" in (res.error or "")
    assert res.execution_duration is not None
