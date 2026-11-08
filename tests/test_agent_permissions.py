"""Unit tests for the SYNQRA Tool Permissions and Approval Validation."""

from datetime import UTC, datetime, timedelta

from synqra.agents.permissions import (
    TOOL_PERMISSIONS,
    TOOLS_REQUIRING_CONFIRMATION,
    ApprovalContext,
    get_tool_permission_level,
    validate_approval_for_execution,
)
from synqra.agents.schemas import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStatus,
    PermissionLevel,
)


def test_tool_permissions_catalog_coverage() -> None:
    """Verify that all 30 expected integration tools have explicit permission tiers."""
    assert len(TOOL_PERMISSIONS) == 30

    # Test sample classifications
    assert get_tool_permission_level("search_web") == PermissionLevel.READ
    assert get_tool_permission_level("search_youtube_videos") == PermissionLevel.READ
    assert get_tool_permission_level("create_google_document") == PermissionLevel.WRITE
    assert get_tool_permission_level("upload_drive_file") == PermissionLevel.WRITE
    assert get_tool_permission_level("send_gmail_message") == PermissionLevel.SENSITIVE
    assert get_tool_permission_level("delete_calendar_event") == PermissionLevel.SENSITIVE

    # Unrecognized tools default to SENSITIVE
    assert get_tool_permission_level("unknown_dangerous_tool") == PermissionLevel.SENSITIVE


def test_read_tools_auto_approved() -> None:
    """Verify READ tools can execute without approval context."""
    is_valid, reason = validate_approval_for_execution(
        tool_name="search_web",
        arguments={"query": "python"},
        permission_level=PermissionLevel.READ,
        workflow_id="wf_123",
        approval_context=None,
    )
    assert is_valid is True
    assert reason is None


def test_write_and_sensitive_tools_require_approval() -> None:
    """Verify WRITE and SENSITIVE tools are blocked if approval_context is missing."""
    for perm in (PermissionLevel.WRITE, PermissionLevel.SENSITIVE):
        is_valid, reason = validate_approval_for_execution(
            tool_name="upload_drive_file",
            arguments={"name": "test.txt", "content": "hello"},
            permission_level=perm,
            workflow_id="wf_123",
            approval_context=None,
        )
        assert is_valid is False
        assert "requires human approval" in (reason or "")


def test_valid_approval_context_authorized() -> None:
    """Verify valid human approval context permits execution."""
    req = ApprovalRequest(
        workflow_id="wf_123",
        tool_name="create_google_document",
        arguments={"title": "Notes", "content": "Research"},
        reason="Testing",
    )
    decision = ApprovalDecision(
        approval_id=req.approval_id,
        approved=True,
        decided_by="admin_alice",
    )
    ctx = ApprovalContext(approval_request=req, approval_decision=decision)

    is_valid, reason = validate_approval_for_execution(
        tool_name="create_google_document",
        arguments={"title": "Notes", "content": "Research"},
        permission_level=PermissionLevel.WRITE,
        workflow_id="wf_123",
        approval_context=ctx,
    )
    assert is_valid is True
    assert reason is None


def test_rejected_approval_blocks_execution() -> None:
    """Verify rejected approval decision blocks execution."""
    req = ApprovalRequest(
        workflow_id="wf_123",
        tool_name="create_calendar_event",
        arguments={"summary": "Meeting"},
        reason="Testing",
    )
    decision = ApprovalDecision(
        approval_id=req.approval_id,
        approved=False,
        decided_by="manager_bob",
        reason="Schedule conflict",
    )
    ctx = ApprovalContext(approval_request=req, approval_decision=decision)

    is_valid, reason = validate_approval_for_execution(
        tool_name="create_calendar_event",
        arguments={"summary": "Meeting"},
        permission_level=PermissionLevel.WRITE,
        workflow_id="wf_123",
        approval_context=ctx,
    )
    assert is_valid is False
    assert "was rejected by manager_bob" in (reason or "")


def test_expired_approval_blocks_execution() -> None:
    """Verify expired approval request is rejected."""
    now = datetime.now(UTC)
    req = ApprovalRequest(
        workflow_id="wf_123",
        tool_name="upload_drive_file",
        arguments={"name": "file.txt"},
        reason="Testing",
        expires_at=now + timedelta(seconds=1),
    )
    decision = ApprovalDecision(
        approval_id=req.approval_id,
        approved=True,
        decided_by="admin_alice",
    )
    ctx = ApprovalContext(approval_request=req, approval_decision=decision)

    # Manually mark expired or test after expiry
    req.expires_at = now - timedelta(seconds=10)
    req.status = ApprovalStatus.EXPIRED

    is_valid, reason = validate_approval_for_execution(
        tool_name="upload_drive_file",
        arguments={"name": "file.txt"},
        permission_level=PermissionLevel.WRITE,
        workflow_id="wf_123",
        approval_context=ctx,
    )
    assert is_valid is False
    assert "has expired" in (reason or "")


def test_workflow_id_mismatch_blocks_execution() -> None:
    """Verify approval issued for workflow A cannot be reused in workflow B."""
    req = ApprovalRequest(
        workflow_id="wf_AAA",
        tool_name="upload_drive_file",
        arguments={"name": "doc.pdf"},
        reason="Testing",
    )
    decision = ApprovalDecision(approval_id=req.approval_id, approved=True, decided_by="admin")
    ctx = ApprovalContext(approval_request=req, approval_decision=decision)

    is_valid, reason = validate_approval_for_execution(
        tool_name="upload_drive_file",
        arguments={"name": "doc.pdf"},
        permission_level=PermissionLevel.WRITE,
        workflow_id="wf_BBB",
        approval_context=ctx,
    )
    assert is_valid is False
    assert "Workflow ID mismatch" in (reason or "")


def test_tool_name_mismatch_blocks_execution() -> None:
    """Verify approval for tool A cannot be used to execute tool B."""
    req = ApprovalRequest(
        workflow_id="wf_123",
        tool_name="create_google_document",
        arguments={"title": "Doc"},
        reason="Testing",
    )
    decision = ApprovalDecision(approval_id=req.approval_id, approved=True, decided_by="admin")
    ctx = ApprovalContext(approval_request=req, approval_decision=decision)

    is_valid, reason = validate_approval_for_execution(
        tool_name="update_google_document",
        arguments={"title": "Doc"},
        permission_level=PermissionLevel.WRITE,
        workflow_id="wf_123",
        approval_context=ctx,
    )
    assert is_valid is False
    assert "Tool mismatch" in (reason or "")


def test_modified_arguments_block_execution() -> None:
    """Verify modifying parameters after approval was granted invalidates the approval."""
    req = ApprovalRequest(
        workflow_id="wf_123",
        tool_name="create_calendar_event",
        arguments={"summary": "Safe Meeting", "duration_minutes": 30},
        reason="Testing",
    )
    decision = ApprovalDecision(approval_id=req.approval_id, approved=True, decided_by="admin")
    ctx = ApprovalContext(approval_request=req, approval_decision=decision)

    # Attacker or agent altered arguments
    altered_args = {"summary": "Safe Meeting", "duration_minutes": 180}
    is_valid, reason = validate_approval_for_execution(
        tool_name="create_calendar_event",
        arguments=altered_args,
        permission_level=PermissionLevel.WRITE,
        workflow_id="wf_123",
        approval_context=ctx,
    )
    assert is_valid is False
    assert "Argument mismatch" in (reason or "")


def test_mandatory_confirmation_protection() -> None:
    """Verify tools requiring confirm=True are blocked if confirm is False or missing."""
    for tool_name in TOOLS_REQUIRING_CONFIRMATION:
        req = ApprovalRequest(
            workflow_id="wf_123",
            tool_name=tool_name,
            arguments={"recipient": "a@b.com", "subject": "Hi", "body": "Msg", "confirm": False},
            reason="Testing",
        )
        decision = ApprovalDecision(approval_id=req.approval_id, approved=True, decided_by="admin")
        ctx = ApprovalContext(approval_request=req, approval_decision=decision)

        is_valid, reason = validate_approval_for_execution(
            tool_name=tool_name,
            arguments={"recipient": "a@b.com", "subject": "Hi", "body": "Msg", "confirm": False},
            permission_level=PermissionLevel.SENSITIVE,
            workflow_id="wf_123",
            approval_context=ctx,
        )
        assert is_valid is False
        assert "requires explicit confirm=True" in (reason or "")
