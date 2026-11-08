"""Contract verification for 31 original curated tools + 1 additive agent tool (32 total).

Ensures the public FastMCP server interface preserves the exact original tool inventory (31 tools),
plus the additive synqra_run_agent tool (32 tools total), canonical names, descriptions,
input schemas, required parameters, and types without exposing internal implementation details
or raw Composio action names.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastmcp import FastMCP

from synqra.core.config import Settings
from synqra.integrations.github import GitHubIntegration
from synqra.integrations.gmail import GmailIntegration
from synqra.integrations.google_calendar import GoogleCalendarIntegration
from synqra.integrations.google_drive import GoogleDriveIntegration
from synqra.integrations.registry import IntegrationRegistry
from synqra.integrations.web_search import WebSearchIntegration
from synqra.integrations.youtube import YouTubeIntegration
from synqra.server.mcp_server import create_mcp_server


@pytest.fixture
def fully_mounted_mcp_server() -> FastMCP:
    """Fixture providing a FastMCP server with all 6 integrations configured and mounted."""
    settings = Settings(
        composio_api_key="mock_composio_key_for_contract_tests",
        tavily_api_key="mock_tavily_key_for_contract_tests",
    )
    reg = IntegrationRegistry()
    reg.register(YouTubeIntegration(settings=settings))
    reg.register(GitHubIntegration(settings=settings))
    reg.register(GoogleDriveIntegration(settings=settings))
    reg.register(GoogleCalendarIntegration(settings=settings))
    reg.register(GmailIntegration(settings=settings))
    reg.register(WebSearchIntegration(settings=settings))
    return create_mcp_server(settings=settings, registry=reg)


# Complete 32-tool specification: 31 original curated tools + 1 additive agent tool
EXPECTED_INVENTORY: dict[str, dict[str, Any]] = {
    # YouTube (2)
    "search_youtube_videos": {
        "service": "youtube",
        "required": ["query"],
        "properties": ["query", "max_results", "order"],
    },
    "get_video_details": {
        "service": "youtube",
        "required": ["video_id"],
        "properties": ["video_id"],
    },
    # GitHub (5)
    "search_github_repositories": {
        "service": "github",
        "required": ["query"],
        "properties": ["query", "language", "sort", "order", "page", "per_page"],
    },
    "get_github_repository": {
        "service": "github",
        "required": ["owner", "repo"],
        "properties": ["owner", "repo"],
    },
    "get_github_file_contents": {
        "service": "github",
        "required": ["owner", "repo", "path"],
        "properties": ["owner", "repo", "path", "ref"],
    },
    "list_github_issues": {
        "service": "github",
        "required": ["owner", "repo"],
        "properties": ["owner", "repo", "state", "labels", "page", "per_page"],
    },
    "get_github_readme": {
        "service": "github",
        "required": ["owner", "repo"],
        "properties": ["owner", "repo", "ref"],
    },
    # Google Drive / Docs (7)
    "search_drive_files": {
        "service": "google_drive",
        "required": [],
        "properties": ["query", "name", "mime_type", "folder_id", "page_size"],
    },
    "get_drive_file": {
        "service": "google_drive",
        "required": ["file_id"],
        "properties": ["file_id"],
    },
    "download_drive_file": {
        "service": "google_drive",
        "required": ["file_id"],
        "properties": ["file_id", "mime_type"],
    },
    "upload_drive_file": {
        "service": "google_drive",
        "required": ["name", "content"],
        "properties": ["name", "content", "mime_type", "parent_id"],
    },
    "create_google_document": {
        "service": "google_drive",
        "required": ["title", "content"],
        "properties": ["title", "content", "parent_id"],
    },
    "update_google_document": {
        "service": "google_drive",
        "required": ["file_id", "content"],
        "properties": ["file_id", "content", "find_text", "insertion_index", "mime_type"],
    },
    "create_drive_folder": {
        "service": "google_drive",
        "required": ["name"],
        "properties": ["name", "parent_id"],
    },
    # Google Calendar (7)
    "list_calendars": {
        "service": "google_calendar",
        "required": [],
        "properties": ["max_results", "page_token", "show_hidden", "min_access_role"],
    },
    "list_calendar_events": {
        "service": "google_calendar",
        "required": [],
        "properties": [
            "calendar_id",
            "time_min",
            "time_max",
            "timezone",
            "query",
            "max_results",
            "page_token",
            "single_events",
            "order_by",
        ],
    },
    "get_calendar_event": {
        "service": "google_calendar",
        "required": ["event_id"],
        "properties": ["event_id", "calendar_id", "timezone"],
    },
    "create_calendar_event": {
        "service": "google_calendar",
        "required": ["summary", "start_time"],
        "properties": [
            "summary",
            "start_time",
            "end_time",
            "duration_minutes",
            "description",
            "location",
            "timezone",
            "calendar_id",
            "attendees",
            "reminders_minutes",
            "is_all_day",
        ],
    },
    "update_calendar_event": {
        "service": "google_calendar",
        "required": ["event_id"],
        "properties": [
            "event_id",
            "calendar_id",
            "summary",
            "description",
            "location",
            "start_time",
            "end_time",
            "timezone",
            "attendees",
        ],
    },
    "delete_calendar_event": {
        "service": "google_calendar",
        "required": ["event_id"],
        "properties": ["event_id", "calendar_id", "confirm"],
    },
    "find_free_time": {
        "service": "google_calendar",
        "required": ["time_min", "time_max"],
        "properties": [
            "time_min",
            "time_max",
            "calendar_ids",
            "duration_minutes",
            "timezone",
        ],
    },
    # Gmail (6)
    "search_gmail_messages": {
        "service": "gmail",
        "required": [],
        "properties": [
            "query",
            "sender",
            "recipient",
            "subject",
            "keywords",
            "after",
            "before",
            "label_ids",
            "max_results",
            "page_token",
            "include_spam_trash",
        ],
    },
    "get_gmail_message": {
        "service": "gmail",
        "required": ["message_id"],
        "properties": ["message_id", "format"],
    },
    "list_gmail_labels": {
        "service": "gmail",
        "required": [],
        "properties": ["include_details"],
    },
    "create_gmail_draft": {
        "service": "gmail",
        "required": ["recipient", "subject", "body"],
        "properties": [
            "recipient",
            "subject",
            "body",
            "cc",
            "bcc",
            "is_html",
            "thread_id",
        ],
    },
    "send_gmail_message": {
        "service": "gmail",
        "required": ["recipient", "subject", "body"],
        "properties": [
            "recipient",
            "subject",
            "body",
            "cc",
            "bcc",
            "is_html",
            "confirm",
        ],
    },
    "reply_to_gmail_message": {
        "service": "gmail",
        "required": ["message_id", "reply_body"],
        "properties": [
            "message_id",
            "reply_body",
            "thread_id",
            "recipient",
            "cc",
            "bcc",
            "is_html",
            "confirm",
        ],
    },
    # Web Search (3)
    "search_web": {
        "service": "web_search",
        "required": ["query"],
        "properties": [
            "query",
            "max_results",
            "search_depth",
            "include_domains",
            "exclude_domains",
        ],
    },
    "extract_webpage_content": {
        "service": "web_search",
        "required": ["url"],
        "properties": ["url"],
    },
    "search_news": {
        "service": "web_search",
        "required": ["query"],
        "properties": [
            "query",
            "max_results",
            "days",
            "include_domains",
            "exclude_domains",
        ],
    },
    # System (1)
    "health_check": {
        "service": "system",
        "required": [],
        "properties": [],
    },
    # Agent Invocation (Phase 9.6 Component B - 1)
    "synqra_run_agent": {
        "service": "agent",
        "required": ["request"],
        "properties": ["request", "thread_id"],
    },
}


@pytest.mark.asyncio
async def test_mcp_exact_31_tool_inventory(fully_mounted_mcp_server: FastMCP) -> None:
    """Verify FastMCP registers exactly 32 tools (31 original + 1 additive agent tool)."""
    tools = await fully_mounted_mcp_server.list_tools()

    tool_names = {t.name for t in tools}
    expected_names = set(EXPECTED_INVENTORY.keys())

    assert len(tools) == 32, (
        f"Expected 32 tools (31 original + synqra_run_agent), found {len(tools)}: {tool_names}"
    )
    assert tool_names == expected_names, (
        f"Tool set mismatch! Missing: {expected_names - tool_names}, "
        f"Extra: {tool_names - expected_names}"
    )


@pytest.mark.asyncio
async def test_mcp_tool_schemas_and_parameter_contracts(
    fully_mounted_mcp_server: FastMCP,
) -> None:
    """Deep verification of each tool's schema, required properties, and descriptions."""
    tools = await fully_mounted_mcp_server.list_tools()
    tool_map = {t.name: t for t in tools}

    for name, spec in EXPECTED_INVENTORY.items():
        assert name in tool_map, f"Expected tool '{name}' not found in registered tools"
        tool = tool_map[name]

        # Verify description exists and is meaningful
        assert tool.description and len(tool.description.strip()) > 10, (
            f"Tool '{name}' has empty or invalid description: {tool.description}"
        )

        # Retrieve JSON schema
        schema = (
            tool.parameters if hasattr(tool, "parameters") else getattr(tool, "inputSchema", {})
        )
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        required = schema.get("required", []) if isinstance(schema, dict) else []

        # Check required fields match
        expected_required = set(spec["required"])
        actual_required = set(required)
        assert actual_required == expected_required, (
            f"Tool '{name}' required params mismatch. "
            f"Expected {expected_required}, got {actual_required}"
        )

        # Check property names match
        expected_properties = set(spec["properties"])
        actual_properties = set(properties.keys())
        assert actual_properties == expected_properties, (
            f"Tool '{name}' properties mismatch. "
            f"Expected {expected_properties}, got {actual_properties}"
        )


@pytest.mark.asyncio
async def test_no_raw_composio_action_names_exposed(
    fully_mounted_mcp_server: FastMCP,
) -> None:
    """Verify that raw uppercase Composio actions are never exposed to MCP clients."""
    tools = await fully_mounted_mcp_server.list_tools()

    for tool in tools:
        # Composio actions are typically UPPERCASE_WITH_UNDERSCORES
        assert not tool.name.isupper(), f"Raw uppercase action exposed: {tool.name}"
        assert not tool.name.startswith("COMPOSIO_"), f"Raw Composio tool exposed: {tool.name}"


@pytest.mark.asyncio
async def test_service_counts(fully_mounted_mcp_server: FastMCP) -> None:
    """Verify tool distribution across services matches the architecture specification."""
    tools = await fully_mounted_mcp_server.list_tools()

    counts: dict[str, int] = {}
    for t in tools:
        spec = EXPECTED_INVENTORY.get(t.name)
        assert spec is not None, f"Unrecognized tool: {t.name}"
        svc = spec["service"]
        counts[svc] = counts.get(svc, 0) + 1

    assert counts.get("youtube") == 2
    assert counts.get("github") == 5
    assert counts.get("google_drive") == 7
    assert counts.get("google_calendar") == 7
    assert counts.get("gmail") == 6
    assert counts.get("web_search") == 3
    assert counts.get("system") == 1
    assert counts.get("agent") == 1
