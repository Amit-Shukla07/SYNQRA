"""Permission classifications and approval validation foundation for SYNQRA tools.

Establishes strict access control tiers (READ, WRITE, SENSITIVE) across all 30
curated integration tools and provides deterministic, verifiable human approval checks.
"""

from typing import Any

from pydantic import BaseModel, Field

from synqra.agents.schemas import (
    DISALLOWED_DECIDED_BY_IDENTIFIERS,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStatus,
    PermissionLevel,
)

# Canonical mapping of all 30 SYNQRA integration tools to their permission tiers
TOOL_PERMISSIONS: dict[str, PermissionLevel] = {
    # YouTube (2 tools - READ)
    "search_youtube_videos": PermissionLevel.READ,
    "get_video_details": PermissionLevel.READ,
    # GitHub (5 tools - READ)
    "search_github_repositories": PermissionLevel.READ,
    "get_github_repository": PermissionLevel.READ,
    "get_github_file_contents": PermissionLevel.READ,
    "list_github_issues": PermissionLevel.READ,
    "get_github_readme": PermissionLevel.READ,
    # Google Drive & Docs (7 tools: 3 READ, 4 WRITE)
    "search_drive_files": PermissionLevel.READ,
    "get_drive_file": PermissionLevel.READ,
    "download_drive_file": PermissionLevel.READ,
    "upload_drive_file": PermissionLevel.WRITE,
    "create_google_document": PermissionLevel.WRITE,
    "update_google_document": PermissionLevel.WRITE,
    "create_drive_folder": PermissionLevel.WRITE,
    # Google Calendar (7 tools: 4 READ, 1 WRITE, 2 SENSITIVE)
    "list_calendars": PermissionLevel.READ,
    "list_calendar_events": PermissionLevel.READ,
    "get_calendar_event": PermissionLevel.READ,
    "find_free_time": PermissionLevel.READ,
    "create_calendar_event": PermissionLevel.WRITE,
    "update_calendar_event": PermissionLevel.SENSITIVE,
    "delete_calendar_event": PermissionLevel.SENSITIVE,
    # Gmail (6 tools: 3 READ, 1 WRITE, 2 SENSITIVE)
    "search_gmail_messages": PermissionLevel.READ,
    "get_gmail_message": PermissionLevel.READ,
    "list_gmail_labels": PermissionLevel.READ,
    "create_gmail_draft": PermissionLevel.WRITE,
    "send_gmail_message": PermissionLevel.SENSITIVE,
    "reply_to_gmail_message": PermissionLevel.SENSITIVE,
    # Web Search (3 tools - READ)
    "search_web": PermissionLevel.READ,
    "extract_webpage_content": PermissionLevel.READ,
    "search_news": PermissionLevel.READ,
}

# Tools requiring explicit confirm=True parameter in their function signature
TOOLS_REQUIRING_CONFIRMATION: frozenset[str] = frozenset(
    {
        "send_gmail_message",
        "reply_to_gmail_message",
        "delete_calendar_event",
    }
)


class ApprovalContext(BaseModel):
    """Verifiable container linking an approval request to a human operator's decision."""

    approval_request: ApprovalRequest = Field(
        description=(
            "The original approval request containing tool name, arguments, and workflow ID."
        )
    )
    approval_decision: ApprovalDecision = Field(
        description="The recorded decision rendered by an authenticated human operator."
    )


def get_tool_permission_level(tool_name: str) -> PermissionLevel:
    """Retrieve the permission classification for a given tool.

    Defaults to SENSITIVE if the tool is not in the recognized catalog to enforce
    defense-in-depth safety.
    """
    return TOOL_PERMISSIONS.get(tool_name, PermissionLevel.SENSITIVE)


def validate_approval_for_execution(
    tool_name: str,
    arguments: dict[str, Any],
    permission_level: PermissionLevel,
    workflow_id: str,
    approval_context: ApprovalContext | None = None,
) -> tuple[bool, str | None]:
    """Validate whether a tool execution is authorized under its permission classification.

    Args:
        tool_name: Name of the tool to be executed.
        arguments: Arguments dictionary supplied for execution.
        permission_level: Permission tier of the tool.
        workflow_id: Identifier of the active workflow.
        approval_context: Optional human approval context.

    Returns:
        A tuple of (is_authorized, rejection_reason). If authorized, reason is None.
    """
    # 1. READ tools execute automatically without human approval
    if permission_level == PermissionLevel.READ:
        return True, None

    # 2. WRITE and SENSITIVE tools require a valid approval context
    if approval_context is None:
        return (
            False,
            f"Execution of {permission_level.value.upper()} tool '{tool_name}' "
            f"requires human approval.",
        )

    req = approval_context.approval_request
    decision = approval_context.approval_decision

    # 3. Check decision outcome
    if not decision.approved:
        reason_msg = f" Reason: {decision.reason}" if decision.reason else ""
        return (
            False,
            f"Execution blocked: approval for tool '{tool_name}' was rejected "
            f"by {decision.decided_by}.{reason_msg}",
        )

    # 4. Check status of request
    if req.status == ApprovalStatus.REJECTED:
        return False, f"Approval request '{req.approval_id}' has been rejected."

    # 5. Check expiration
    if req.is_expired():
        return False, f"Approval request '{req.approval_id}' has expired."

    # 6. Verify human decision-maker (cannot be an automated agent)
    decider = decision.decided_by.strip().lower()
    if decider in DISALLOWED_DECIDED_BY_IDENTIFIERS:
        return (
            False,
            f"Invalid approval: decision maker '{decision.decided_by}' is an automated agent. "
            f"The LLM must never generate or approve its own approval decision.",
        )

    # 7. Verify workflow identity
    if req.workflow_id != workflow_id:
        return (
            False,
            f"Workflow ID mismatch: approval was issued for workflow '{req.workflow_id}', "
            f"but current workflow is '{workflow_id}'.",
        )

    # 8. Verify target tool name
    if req.tool_name != tool_name:
        return (
            False,
            f"Tool mismatch: approval was issued for tool '{req.tool_name}', "
            f"but attempted execution was for '{tool_name}'.",
        )

    # 9. Verify arguments: execution arguments must match approved arguments exactly
    if req.arguments != arguments:
        return (
            False,
            f"Argument mismatch: arguments have been modified since approval was granted for "
            f"tool '{tool_name}'. Approved: {req.arguments}, Attempted: {arguments}.",
        )

    # 10. Check confirmation parameter for safety-critical tools
    if tool_name in TOOLS_REQUIRING_CONFIRMATION and arguments.get("confirm") is not True:
        return (
            False,
            f"Tool '{tool_name}' requires explicit confirm=True in execution arguments.",
        )

    return True, None
