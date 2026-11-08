"""Centralized Agent Tool Discovery, Registration, and Execution Engine.

Binds all 30 curated integration tools across YouTube, GitHub, Google Drive,
Google Calendar, Gmail, and Web Search into a unified, permission-aware execution engine.
"""

import logging
import time
import uuid
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from synqra.agents.permissions import (
    TOOLS_REQUIRING_CONFIRMATION,
    ApprovalContext,
    get_tool_permission_level,
    validate_approval_for_execution,
)
from synqra.agents.schemas import PermissionLevel, ToolExecutionResult, utc_now
from synqra.agents.tool_adapter import AgentToolAdapter
from synqra.agents.tool_registry import AgentToolRegistry, RegisteredTool
from synqra.integrations import IntegrationRegistry, default_registry
from synqra.integrations.exceptions import (
    IntegrationConfigurationError,
    IntegrationExecutionError,
    IntegrationNotFoundError,
)
from synqra.integrations.github.schemas import (
    GitHubGetFileContentsInput,
    GitHubGetReadmeInput,
    GitHubGetRepositoryInput,
    GitHubListIssuesInput,
    GitHubSearchRepositoriesInput,
)
from synqra.integrations.github.tools import (
    get_github_file_contents,
    get_github_readme,
    get_github_repository,
    list_github_issues,
    search_github_repositories,
)
from synqra.integrations.gmail.schemas import (
    CreateGmailDraftInput,
    GetGmailMessageInput,
    ListGmailLabelsInput,
    ReplyToGmailMessageInput,
    SearchGmailMessagesInput,
    SendGmailMessageInput,
)
from synqra.integrations.gmail.tools import (
    create_gmail_draft,
    get_gmail_message,
    list_gmail_labels,
    reply_to_gmail_message,
    search_gmail_messages,
    send_gmail_message,
)
from synqra.integrations.google_calendar.schemas import (
    CreateCalendarEventInput,
    DeleteCalendarEventInput,
    FindFreeTimeInput,
    GetCalendarEventInput,
    ListCalendarEventsInput,
    ListCalendarsInput,
    UpdateCalendarEventInput,
)
from synqra.integrations.google_calendar.tools import (
    create_calendar_event,
    delete_calendar_event,
    find_free_time,
    get_calendar_event,
    list_calendar_events,
    list_calendars,
    update_calendar_event,
)
from synqra.integrations.google_drive.schemas import (
    CreateDriveFolderInput,
    CreateGoogleDocumentInput,
    DownloadDriveFileInput,
    GetDriveFileInput,
    SearchDriveFilesInput,
    UpdateGoogleDocumentInput,
    UploadDriveFileInput,
)
from synqra.integrations.google_drive.tools import (
    create_drive_folder,
    create_google_document,
    download_drive_file,
    get_drive_file,
    search_drive_files,
    update_google_document,
    upload_drive_file,
)
from synqra.integrations.web_search.schemas import (
    ExtractWebpageInput,
    SearchNewsInput,
    SearchWebInput,
)
from synqra.integrations.web_search.tools import (
    extract_webpage_content,
    search_news,
    search_web,
)
from synqra.integrations.youtube.schemas import (
    YouTubeSearchInput,
    YouTubeVideoDetailsInput,
)
from synqra.integrations.youtube.tools import (
    get_video_details,
    search_youtube_videos,
)

logger = logging.getLogger("synqra.agents.tools")

# Canonical specification for all 30 SYNQRA integration tools
TOOL_DEFINITIONS: list[dict[str, Any]] = [
    # YouTube (2 tools)
    {
        "name": "search_youtube_videos",
        "integration": "youtube",
        "description": (
            "Search YouTube for educational videos, lecture recordings, tutorials, "
            "and learning materials. Returns video titles, watch URLs, channel names, "
            "and descriptions."
        ),
        "schema": YouTubeSearchInput,
        "function": search_youtube_videos,
    },
    {
        "name": "get_video_details",
        "integration": "youtube",
        "description": (
            "Retrieve detailed metadata for a YouTube video by its video ID, including "
            "title, full description, channel name, publication date, duration, and view count."
        ),
        "schema": YouTubeVideoDetailsInput,
        "function": get_video_details,
    },
    # GitHub (5 tools)
    {
        "name": "search_github_repositories",
        "integration": "github",
        "description": (
            "Search GitHub repositories by keywords, language, and sorting filters. "
            "Supports pagination and configurable result limits."
        ),
        "schema": GitHubSearchRepositoriesInput,
        "function": search_github_repositories,
    },
    {
        "name": "get_github_repository",
        "integration": "github",
        "description": (
            "Retrieve detailed information and metrics for an individual GitHub repository, "
            "including description, stars, forks, language, topics, creation and update dates."
        ),
        "schema": GitHubGetRepositoryInput,
        "function": get_github_repository,
    },
    {
        "name": "get_github_file_contents",
        "integration": "github",
        "description": (
            "Retrieve file content from a repository safely. Supports branch/tag/commit "
            "selection and handles binary and text files appropriately."
        ),
        "schema": GitHubGetFileContentsInput,
        "function": get_github_file_contents,
    },
    {
        "name": "list_github_issues",
        "integration": "github",
        "description": (
            "Retrieve repository issues filtered by state and labels, strictly excluding "
            "pull requests. Supports pagination."
        ),
        "schema": GitHubListIssuesInput,
        "function": list_github_issues,
    },
    {
        "name": "get_github_readme",
        "integration": "github",
        "description": "Retrieve the normalized README content and metadata for a repository.",
        "schema": GitHubGetReadmeInput,
        "function": get_github_readme,
    },
    # Google Drive & Docs (7 tools)
    {
        "name": "search_drive_files",
        "integration": "google_drive",
        "description": (
            "Search files and folders in Google Drive by name, MIME type, and query filters. "
            "Supports pagination and folder scoping."
        ),
        "schema": SearchDriveFilesInput,
        "function": search_drive_files,
    },
    {
        "name": "get_drive_file",
        "integration": "google_drive",
        "description": (
            "Retrieve detailed metadata for a Google Drive file or folder by ID, including "
            "name, MIME type, size, modification timestamp, web view URL, and parent folders."
        ),
        "schema": GetDriveFileInput,
        "function": get_drive_file,
    },
    {
        "name": "download_drive_file",
        "integration": "google_drive",
        "description": (
            "Retrieve file contents from Google Drive. Handles text files, binary files, "
            "and Google Workspace document exports (Docs, Sheets) safely."
        ),
        "schema": DownloadDriveFileInput,
        "function": download_drive_file,
    },
    {
        "name": "upload_drive_file",
        "integration": "google_drive",
        "description": (
            "Upload a text or structured file to Google Drive with configurable name, "
            "MIME type, and optional target parent folder."
        ),
        "schema": UploadDriveFileInput,
        "function": upload_drive_file,
    },
    {
        "name": "create_google_document",
        "integration": "google_drive",
        "description": (
            "Create a Google Docs document with supplied title and initial text content, "
            "optionally placed in a target folder."
        ),
        "schema": CreateGoogleDocumentInput,
        "function": create_google_document,
    },
    {
        "name": "update_google_document",
        "integration": "google_drive",
        "description": (
            "Update an existing Google Docs document or Drive file. Supports inserting text "
            "(at end or specific index) and replacing all occurrences of text."
        ),
        "schema": UpdateGoogleDocumentInput,
        "function": update_google_document,
    },
    {
        "name": "create_drive_folder",
        "integration": "google_drive",
        "description": (
            "Create a folder inside Google Drive, optionally under a specified parent folder."
        ),
        "schema": CreateDriveFolderInput,
        "function": create_drive_folder,
    },
    # Google Calendar (7 tools)
    {
        "name": "list_calendars",
        "integration": "google_calendar",
        "description": (
            "List Google Calendars accessible to the connected user, including primary "
            "and secondary calendars with access roles and timezone configurations."
        ),
        "schema": ListCalendarsInput,
        "function": list_calendars,
    },
    {
        "name": "list_calendar_events",
        "integration": "google_calendar",
        "description": (
            "Retrieve calendar events within a specified date/time range. Supports pagination, "
            "timezone-aware formatting, single event expansion, and text queries."
        ),
        "schema": ListCalendarEventsInput,
        "function": list_calendar_events,
    },
    {
        "name": "get_calendar_event",
        "integration": "google_calendar",
        "description": (
            "Retrieve complete details for a specific calendar event using "
            "calendar ID and event ID. Includes attendees, descriptions, "
            "locations, status, reminders, and start/end times."
        ),
        "schema": GetCalendarEventInput,
        "function": get_calendar_event,
    },
    {
        "name": "create_calendar_event",
        "integration": "google_calendar",
        "description": (
            "Create a new timed or all-day Google Calendar event with title, "
            "description, location, timezone, start/end times, and optional attendees."
        ),
        "schema": CreateCalendarEventInput,
        "function": create_calendar_event,
    },
    {
        "name": "update_calendar_event",
        "integration": "google_calendar",
        "description": (
            "Update selected fields of an existing calendar event without "
            "unintentionally overwriting other fields (patch semantics). "
            "Supports title, description, location, start/end times, "
            "timezone, and attendees."
        ),
        "schema": UpdateCalendarEventInput,
        "function": update_calendar_event,
    },
    {
        "name": "delete_calendar_event",
        "integration": "google_calendar",
        "description": (
            "Delete a calendar event from Google Calendar. For safety, this destructive "
            "operation requires explicit confirmation (confirm=True). If confirm is False, "
            "the event will not be deleted."
        ),
        "schema": DeleteCalendarEventInput,
        "function": delete_calendar_event,
    },
    {
        "name": "find_free_time",
        "integration": "google_calendar",
        "description": (
            "Identify available free time slots within a specified date/time range. "
            "Accounts for existing events, filters by requested minimum meeting duration, "
            "and supports multiple calendar IDs."
        ),
        "schema": FindFreeTimeInput,
        "function": find_free_time,
    },
    # Gmail (6 tools)
    {
        "name": "search_gmail_messages",
        "integration": "gmail",
        "description": (
            "Search Gmail messages using text queries, sender/recipient filters, "
            "subject lines, date ranges, and labels. Returns normalized summaries."
        ),
        "schema": SearchGmailMessagesInput,
        "function": search_gmail_messages,
    },
    {
        "name": "get_gmail_message",
        "integration": "gmail",
        "description": (
            "Retrieve complete details for a specific Gmail email using its message ID. "
            "Includes headers, plain text and HTML bodies, and attachment metadata."
        ),
        "schema": GetGmailMessageInput,
        "function": get_gmail_message,
    },
    {
        "name": "list_gmail_labels",
        "integration": "gmail",
        "description": (
            "List all system and user-defined Gmail labels in the user's mailbox, "
            "including unread counts and total message counts."
        ),
        "schema": ListGmailLabelsInput,
        "function": list_gmail_labels,
    },
    {
        "name": "create_gmail_draft",
        "integration": "gmail",
        "description": (
            "Create a new email draft in Gmail without sending it. Supports recipients, "
            "CC, BCC, subject line, plain text or HTML bodies, and threading."
        ),
        "schema": CreateGmailDraftInput,
        "function": create_gmail_draft,
    },
    {
        "name": "send_gmail_message",
        "integration": "gmail",
        "description": (
            "Send an email immediately via Gmail. For safety, this external transmission "
            "requires explicit confirmation (confirm=True). If confirm is False, the email "
            "will not be sent."
        ),
        "schema": SendGmailMessageInput,
        "function": send_gmail_message,
    },
    {
        "name": "reply_to_gmail_message",
        "integration": "gmail",
        "description": (
            "Reply to an existing email message, preserving thread context and headers. "
            "For safety, sending a reply requires explicit confirmation (confirm=True)."
        ),
        "schema": ReplyToGmailMessageInput,
        "function": reply_to_gmail_message,
    },
    # Web Search (3 tools)
    {
        "name": "search_web",
        "integration": "web_search",
        "description": (
            "Search the web for technical documentation, tutorials, articles, and guides. "
            "Supports configurable result limits, search depth, and domain filtering."
        ),
        "schema": SearchWebInput,
        "function": search_web,
    },
    {
        "name": "extract_webpage_content",
        "integration": "web_search",
        "description": (
            "Retrieve and extract clean, readable text or markdown content from a public "
            "webpage URL. Guards against SSRF and internal network destinations."
        ),
        "schema": ExtractWebpageInput,
        "function": extract_webpage_content,
    },
    {
        "name": "search_news",
        "integration": "web_search",
        "description": (
            "Search recent news articles by topic or query, filtered by recency window "
            "(days) and domain restrictions. Returns headlines and summaries."
        ),
        "schema": SearchNewsInput,
        "function": search_news,
    },
]


def _make_client_executor(
    integration_name: str,
    tool_fn: Callable[..., Any],
    integration_registry: IntegrationRegistry,
) -> Callable[..., Any]:
    """Create a client-injected execution wrapper that retrieves the active integration client."""

    async def _executor(**kwargs: Any) -> Any:
        integration = integration_registry.get(integration_name)
        if integration is None:
            raise IntegrationNotFoundError(integration_name)

        if not integration.is_configured():
            raise IntegrationConfigurationError(
                f"Integration '{integration_name}' is not configured. "
                f"Ensure required credentials are provided.",
                integration_name=integration_name,
            )

        client = getattr(integration, "client", None)
        if client is None:
            raise IntegrationExecutionError(
                f"Client for integration '{integration_name}' is unavailable.",
                integration_name=integration_name,
            )

        return await tool_fn(client=client, **kwargs)

    return _executor


def build_agent_tool_registry(
    integration_registry: IntegrationRegistry | None = None,
) -> AgentToolRegistry:
    """Construct an AgentToolRegistry containing all 30 curated integration tools.

    Args:
        integration_registry: Optional IntegrationRegistry source. Defaults to default_registry.

    Returns:
        Configured AgentToolRegistry populated with all 30 tools.
    """
    reg = integration_registry or default_registry
    agent_registry = AgentToolRegistry()

    for item in TOOL_DEFINITIONS:
        name = item["name"]
        integration_name = item["integration"]
        desc = item["description"]
        schema = item["schema"]
        fn = item["function"]
        perm = get_tool_permission_level(name)

        executor = _make_client_executor(
            integration_name=integration_name,
            tool_fn=fn,
            integration_registry=reg,
        )

        tool = RegisteredTool(
            name=name,
            description=desc,
            integration_name=integration_name,
            input_schema=schema,
            execution_function=executor,
            permission_level=perm,
            is_active=True,
            is_async=True,
        )
        agent_registry.register_tool(tool)

    return agent_registry


_DEFAULT_AGENT_REGISTRY: AgentToolRegistry | None = None
_DEFAULT_ADAPTER: AgentToolAdapter | None = None


def get_default_agent_tool_registry() -> AgentToolRegistry:
    """Retrieve or initialize the global singleton AgentToolRegistry."""
    global _DEFAULT_AGENT_REGISTRY
    if _DEFAULT_AGENT_REGISTRY is None:
        _DEFAULT_AGENT_REGISTRY = build_agent_tool_registry()
    return _DEFAULT_AGENT_REGISTRY


def get_default_tool_adapter() -> AgentToolAdapter:
    """Retrieve or initialize the global singleton AgentToolAdapter."""
    global _DEFAULT_ADAPTER
    if _DEFAULT_ADAPTER is None:
        _DEFAULT_ADAPTER = AgentToolAdapter()
    return _DEFAULT_ADAPTER


async def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    workflow_id: str,
    approval_context: ApprovalContext | None = None,
    registry: AgentToolRegistry | None = None,
    adapter: AgentToolAdapter | None = None,
    timeout: float = 60.0,
) -> ToolExecutionResult:
    """Execute a SYNQRA tool with complete lifecycle validation, permissions, and duration tracking.

    Args:
        tool_name: Name of the tool to invoke.
        arguments: Arguments dictionary to validate and supply to the tool.
        workflow_id: Unique identifier of the current execution workflow.
        approval_context: Optional human approval context for WRITE/SENSITIVE tools.
        registry: AgentToolRegistry to query (defaults to global singleton).
        adapter: AgentToolAdapter to execute through (defaults to global singleton).
        timeout: Execution timeout limit in seconds.

    Returns:
        Structured ToolExecutionResult recording success, outputs, errors, and timing.
    """
    exec_id = f"exec_{uuid.uuid4().hex[:12]}"
    started_at = utc_now()
    start_mono = time.monotonic()

    tool_reg = registry or get_default_agent_tool_registry()
    tool_adapter = adapter or get_default_tool_adapter()

    # 1. Verify tool exists in registry
    tool = tool_reg.get_tool(tool_name)
    if tool is None:
        duration = time.monotonic() - start_mono
        return ToolExecutionResult(
            execution_id=exec_id,
            tool_name=tool_name,
            success=False,
            error=f"Tool '{tool_name}' is not registered in the agent tool registry.",
            execution_duration=duration,
            started_at=started_at,
            completed_at=utc_now(),
        )

    # 2. Verify tool is currently active
    if not tool.is_active:
        duration = time.monotonic() - start_mono
        return ToolExecutionResult(
            execution_id=exec_id,
            tool_name=tool_name,
            success=False,
            error=f"Tool '{tool_name}' is currently disabled in the agent tool registry.",
            execution_duration=duration,
            started_at=started_at,
            completed_at=utc_now(),
        )

    # 3. Check permission classification and validate approval
    is_authorized, rejection_reason = validate_approval_for_execution(
        tool_name=tool_name,
        arguments=arguments,
        permission_level=tool.permission_level,
        workflow_id=workflow_id,
        approval_context=approval_context,
    )
    if not is_authorized:
        duration = time.monotonic() - start_mono
        return ToolExecutionResult(
            execution_id=exec_id,
            tool_name=tool_name,
            success=False,
            error=rejection_reason or "Tool execution rejected by permission policy.",
            execution_duration=duration,
            started_at=started_at,
            completed_at=utc_now(),
        )

    # 4. Enforce mandatory confirmation for destructive/external tools
    if tool_name in TOOLS_REQUIRING_CONFIRMATION and arguments.get("confirm") is not True:
        duration = time.monotonic() - start_mono
        return ToolExecutionResult(
            execution_id=exec_id,
            tool_name=tool_name,
            success=False,
            error=(
                f"Tool '{tool_name}' requires explicit confirm=True in execution arguments "
                f"for safety confirmation."
            ),
            execution_duration=duration,
            started_at=started_at,
            completed_at=utc_now(),
        )

    # 5. Execute underlying tool via adapter
    try:
        raw_result = await tool_adapter.execute(
            tool=tool,
            arguments=arguments,
            timeout=timeout,
        )
        duration = time.monotonic() - start_mono
        completed_at = utc_now()

        # Normalize Pydantic models to JSON-safe dictionaries
        normalized_output = (
            raw_result.model_dump(mode="json") if hasattr(raw_result, "model_dump") else raw_result
        )

        return ToolExecutionResult(
            execution_id=exec_id,
            tool_name=tool_name,
            success=True,
            output=normalized_output,
            execution_duration=duration,
            started_at=started_at,
            completed_at=completed_at,
        )

    except Exception as exc:
        duration = time.monotonic() - start_mono
        completed_at = utc_now()
        logger.warning("Execution error in tool '%s': %s", tool_name, exc)
        return ToolExecutionResult(
            execution_id=exec_id,
            tool_name=tool_name,
            success=False,
            error=str(exc),
            execution_duration=duration,
            started_at=started_at,
            completed_at=completed_at,
        )


# Discovery helpers for planner and router nodes
def list_tools(active_only: bool = True) -> list[RegisteredTool]:
    """List tools currently available in the default agent tool registry."""
    return get_default_agent_tool_registry().list_tools(active_only=active_only)


def get_tool(name: str) -> RegisteredTool | None:
    """Retrieve tool metadata by name from the default agent tool registry."""
    return get_default_agent_tool_registry().get_tool(name)


def get_tools_by_integration(
    integration_name: str,
    active_only: bool = True,
) -> list[RegisteredTool]:
    """Retrieve tools for a specific integration from the default agent tool registry."""
    return get_default_agent_tool_registry().get_tools_by_integration(
        integration_name=integration_name,
        active_only=active_only,
    )


def get_tools_by_permission(
    permission_level: PermissionLevel,
    active_only: bool = True,
) -> list[RegisteredTool]:
    """Retrieve tools belonging to a permission tier from the default agent tool registry."""
    return get_default_agent_tool_registry().get_tools_by_permission(
        permission_level=permission_level,
        active_only=active_only,
    )


def get_tool_schema(name: str) -> type[BaseModel]:
    """Retrieve the Pydantic input schema for a tool from the default agent tool registry."""
    return get_default_agent_tool_registry().get_tool_schema(name)
