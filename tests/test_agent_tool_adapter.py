"""Unit tests for the SYNQRA AgentToolAdapter."""

import asyncio

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from synqra.agents.schemas import PermissionLevel
from synqra.agents.tool_adapter import AgentToolAdapter
from synqra.agents.tool_registry import RegisteredTool
from synqra.integrations.exceptions import (
    IntegrationConfigurationError,
    IntegrationExecutionError,
)


class SampleInput(BaseModel):
    query: str = Field(description="Search term", min_length=2)
    limit: int = Field(default=5, ge=1, le=20)


class SampleOutput(BaseModel):
    results: list[str]


async def sample_async_func(query: str, limit: int = 5) -> SampleOutput:
    return SampleOutput(results=[f"{query}_{i}" for i in range(limit)])


def sample_sync_func(query: str, limit: int = 5) -> dict[str, str]:
    return {"status": "ok", "query": query, "limit": str(limit)}


async def sample_slow_func(query: str, limit: int = 5) -> str:
    await asyncio.sleep(0.5)
    return f"slow_{query}_{limit}"


async def sample_failing_func(query: str, limit: int = 5) -> str:
    _ = (query, limit)
    raise IntegrationConfigurationError("Config missing", integration_name="mock_svc")


async def sample_unexpected_error_func(query: str, limit: int = 5) -> str:
    _ = (query, limit)
    raise ValueError("Zero division or unexpected bug")


@pytest.mark.asyncio
async def test_adapter_async_function_execution() -> None:
    """Verify executing an asynchronous tool function via the adapter."""
    adapter = AgentToolAdapter()
    tool = RegisteredTool(
        name="sample_async",
        description="Async tool",
        integration_name="mock",
        input_schema=SampleInput,
        execution_function=sample_async_func,
        permission_level=PermissionLevel.READ,
    )

    out = await adapter.execute(tool, {"query": "test", "limit": 2})
    assert isinstance(out, SampleOutput)
    assert out.results == ["test_0", "test_1"]


@pytest.mark.asyncio
async def test_adapter_sync_function_execution() -> None:
    """Verify executing a synchronous tool function in worker thread."""
    adapter = AgentToolAdapter()
    tool = RegisteredTool(
        name="sample_sync",
        description="Sync tool",
        integration_name="mock",
        input_schema=SampleInput,
        execution_function=sample_sync_func,
        permission_level=PermissionLevel.READ,
        is_async=False,
    )

    out = await adapter.execute(tool, {"query": "sync_test", "limit": 3})
    assert out["status"] == "ok"
    assert out["query"] == "sync_test"


@pytest.mark.asyncio
async def test_adapter_input_validation_failure() -> None:
    """Verify that arguments failing Pydantic schema validation raise IntegrationExecutionError."""
    adapter = AgentToolAdapter()
    tool = RegisteredTool(
        name="sample_val",
        description="Validation tool",
        integration_name="mock",
        input_schema=SampleInput,
        execution_function=sample_async_func,
        permission_level=PermissionLevel.READ,
    )

    # query is too short (min_length=2) and limit is invalid (> 20)
    with pytest.raises(IntegrationExecutionError, match="Invalid arguments for tool 'sample_val'"):
        await adapter.execute(tool, {"query": "x", "limit": 100})


@pytest.mark.asyncio
async def test_adapter_timeout_handling() -> None:
    """Verify that functions exceeding the timeout threshold are interrupted."""
    adapter = AgentToolAdapter(default_timeout=0.05)
    tool = RegisteredTool(
        name="sample_slow",
        description="Slow tool",
        integration_name="mock",
        input_schema=SampleInput,
        execution_function=sample_slow_func,
        permission_level=PermissionLevel.READ,
    )

    with pytest.raises(IntegrationExecutionError, match="timed out after 0.1 seconds"):
        await adapter.execute(tool, {"query": "delay"}, timeout=0.05)


@pytest.mark.asyncio
async def test_adapter_preserves_integration_exceptions() -> None:
    """Verify that typed SynqraIntegrationError subclasses are preserved as-is."""
    adapter = AgentToolAdapter()
    tool = RegisteredTool(
        name="sample_fail",
        description="Fail tool",
        integration_name="mock_svc",
        input_schema=SampleInput,
        execution_function=sample_failing_func,
        permission_level=PermissionLevel.READ,
    )

    with pytest.raises(IntegrationConfigurationError, match="Config missing") as exc_info:
        await adapter.execute(tool, {"query": "test"})

    assert exc_info.value.integration_name == "mock_svc"


@pytest.mark.asyncio
async def test_adapter_wraps_unexpected_exceptions() -> None:
    """Verify that unhandled exceptions are normalized into IntegrationExecutionError."""
    adapter = AgentToolAdapter()
    tool = RegisteredTool(
        name="sample_bug",
        description="Buggy tool",
        integration_name="mock_svc",
        input_schema=SampleInput,
        execution_function=sample_unexpected_error_func,
        permission_level=PermissionLevel.READ,
    )

    with pytest.raises(IntegrationExecutionError, match="Error executing tool 'sample_bug'"):
        await adapter.execute(tool, {"query": "test"})


@pytest.mark.asyncio
async def test_adapter_to_structured_tool_langchain() -> None:
    """Verify conversion to LangChain StructuredTool and calling via ainvoke."""
    adapter = AgentToolAdapter()
    tool = RegisteredTool(
        name="lc_tool",
        description="LangChain tool description",
        integration_name="mock",
        input_schema=SampleInput,
        execution_function=sample_async_func,
        permission_level=PermissionLevel.READ,
    )

    st = adapter.to_structured_tool(tool)
    assert isinstance(st, StructuredTool)
    assert st.name == "lc_tool"
    assert st.description == "LangChain tool description"
    assert st.args_schema is SampleInput

    res = await st.ainvoke({"query": "langchain", "limit": 2})
    assert isinstance(res, SampleOutput)
    assert len(res.results) == 2
