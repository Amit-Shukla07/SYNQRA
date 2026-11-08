"""Unit tests for the SYNQRA AgentToolRegistry."""

import pytest
from pydantic import BaseModel, Field

from synqra.agents.exceptions import SynqraAgentError
from synqra.agents.schemas import PermissionLevel
from synqra.agents.tool_registry import AgentToolRegistry, RegisteredTool
from synqra.agents.tools import build_agent_tool_registry, list_tools


class DummyInput(BaseModel):
    query: str = Field(description="Query string")


async def dummy_fn(query: str) -> str:
    return f"result: {query}"


def test_tool_registry_registration_and_lookup() -> None:
    """Verify basic tool registration and lookup."""
    registry = AgentToolRegistry()
    tool = RegisteredTool(
        name="test_tool",
        description="A test tool",
        integration_name="test_integration",
        input_schema=DummyInput,
        execution_function=dummy_fn,
        permission_level=PermissionLevel.READ,
    )

    registry.register_tool(tool)
    assert registry.validate_tool_name("test_tool") is True
    assert registry.get_tool("test_tool") is tool
    assert registry.get_tool_description("test_tool") == "A test tool"
    assert registry.get_tool_schema("test_tool") is DummyInput
    assert registry.get_tool("non_existent") is None


def test_tool_registry_duplicate_registration_rejected() -> None:
    """Verify that registering duplicate tool names raises ValueError."""
    registry = AgentToolRegistry()
    tool1 = RegisteredTool(
        name="duplicate_tool",
        description="Tool 1",
        integration_name="test",
        input_schema=DummyInput,
        execution_function=dummy_fn,
        permission_level=PermissionLevel.READ,
    )
    tool2 = RegisteredTool(
        name="duplicate_tool",
        description="Tool 2",
        integration_name="test",
        input_schema=DummyInput,
        execution_function=dummy_fn,
        permission_level=PermissionLevel.WRITE,
    )

    registry.register_tool(tool1)
    with pytest.raises(ValueError, match="already registered"):
        registry.register_tool(tool2)


def test_tool_registry_get_or_raise() -> None:
    """Verify get_tool_or_raise raises SynqraAgentError on missing tool."""
    registry = AgentToolRegistry()
    with pytest.raises(SynqraAgentError, match="not registered"):
        registry.get_tool_or_raise("missing_tool")


def test_tool_registry_enable_disable() -> None:
    """Verify enabling, disabling, and active-only filtering of tools."""
    registry = AgentToolRegistry()
    tool = RegisteredTool(
        name="toggle_tool",
        description="Toggle tool",
        integration_name="test",
        input_schema=DummyInput,
        execution_function=dummy_fn,
        permission_level=PermissionLevel.READ,
    )
    registry.register_tool(tool)

    assert tool.is_active is True
    assert len(registry.list_tools(active_only=True)) == 1

    registry.disable_tool("toggle_tool")
    assert tool.is_active is False
    assert len(registry.list_tools(active_only=True)) == 0
    assert len(registry.list_tools(active_only=False)) == 1

    registry.enable_tool("toggle_tool")
    assert tool.is_active is True
    assert len(registry.list_tools(active_only=True)) == 1


def test_tool_registry_filters() -> None:
    """Verify filtering tools by integration and permission level."""
    registry = AgentToolRegistry()
    t1 = RegisteredTool(
        name="yt_search",
        description="YT Search",
        integration_name="youtube",
        input_schema=DummyInput,
        execution_function=dummy_fn,
        permission_level=PermissionLevel.READ,
    )
    t2 = RegisteredTool(
        name="cal_create",
        description="Cal Create",
        integration_name="google_calendar",
        input_schema=DummyInput,
        execution_function=dummy_fn,
        permission_level=PermissionLevel.WRITE,
    )
    t3 = RegisteredTool(
        name="gmail_send",
        description="Gmail Send",
        integration_name="gmail",
        input_schema=DummyInput,
        execution_function=dummy_fn,
        permission_level=PermissionLevel.SENSITIVE,
    )

    registry.register_tool(t1)
    registry.register_tool(t2)
    registry.register_tool(t3)

    # Integration filter
    yt_tools = registry.get_tools_by_integration("youtube")
    assert len(yt_tools) == 1
    assert yt_tools[0].name == "yt_search"

    # Permission filter
    read_tools = registry.get_tools_by_permission(PermissionLevel.READ)
    assert len(read_tools) == 1
    assert read_tools[0].name == "yt_search"

    write_tools = registry.get_tools_by_permission(PermissionLevel.WRITE)
    assert len(write_tools) == 1
    assert write_tools[0].name == "cal_create"

    sensitive_tools = registry.get_tools_by_permission(PermissionLevel.SENSITIVE)
    assert len(sensitive_tools) == 1
    assert sensitive_tools[0].name == "gmail_send"


def test_complete_registration_of_all_30_tools() -> None:
    """Verify that build_agent_tool_registry discovers and registers all 30 expected tools."""
    registry = build_agent_tool_registry()
    all_tools = registry.list_tools(active_only=False)
    assert len(all_tools) == 30

    names = {t.name for t in all_tools}

    # YouTube (2)
    assert {"search_youtube_videos", "get_video_details"}.issubset(names)

    # GitHub (5)
    assert {
        "search_github_repositories",
        "get_github_repository",
        "get_github_file_contents",
        "list_github_issues",
        "get_github_readme",
    }.issubset(names)

    # Google Drive (7)
    assert {
        "search_drive_files",
        "get_drive_file",
        "download_drive_file",
        "upload_drive_file",
        "create_google_document",
        "update_google_document",
        "create_drive_folder",
    }.issubset(names)

    # Google Calendar (7)
    assert {
        "list_calendars",
        "list_calendar_events",
        "get_calendar_event",
        "create_calendar_event",
        "update_calendar_event",
        "delete_calendar_event",
        "find_free_time",
    }.issubset(names)

    # Gmail (6)
    assert {
        "search_gmail_messages",
        "get_gmail_message",
        "list_gmail_labels",
        "create_gmail_draft",
        "send_gmail_message",
        "reply_to_gmail_message",
    }.issubset(names)

    # Web Search (3)
    assert {
        "search_web",
        "extract_webpage_content",
        "search_news",
    }.issubset(names)


def test_module_level_list_tools() -> None:
    """Verify module-level discovery functions."""
    tools = list_tools()
    assert len(tools) == 30
