"""Unit tests for SYNQRA approval data models and lifecycle constraints."""

from datetime import UTC, datetime, timedelta

import pytest

from synqra.agents.approval.models import (
    ApprovalDecisionRecord,
    ApprovalDecisionRequest,
    ApprovalListResponse,
    ApprovalRequestRecord,
    ApprovalResponse,
    utc_now,
)
from synqra.agents.exceptions import ApprovalError
from synqra.agents.schemas import ApprovalStatus, PermissionLevel


def test_approval_decision_valid_human() -> None:
    """Verify ApprovalDecisionRecord accepts valid human operators."""
    decision = ApprovalDecisionRecord(
        approval_id="appr_12345",
        approved=True,
        decided_by="admin_alice",
        reason="Verified parameters look correct.",
    )
    assert decision.approval_id == "appr_12345"
    assert decision.approved is True
    assert decision.decided_by == "admin_alice"
    assert decision.reason == "Verified parameters look correct."
    assert decision.decision_id.startswith("dec_")


@pytest.mark.parametrize(
    "disallowed_actor",
    ["llm", "ai", "agent", "assistant", "system", "gpt-4o", "claude", "bot"],
)
def test_approval_decision_rejects_automated_agents(disallowed_actor: str) -> None:
    """Verify that automated agents, bots, or LLM identifiers are strictly disallowed."""
    with pytest.raises(ApprovalError, match="must never approve"):
        ApprovalDecisionRecord(
            approval_id="appr_123",
            approved=True,
            decided_by=disallowed_actor,
        )


def test_approval_decision_rejects_empty_decider() -> None:
    """Verify that empty or whitespace-only decider identifiers are rejected."""
    with pytest.raises(ApprovalError, match="non-empty"):
        ApprovalDecisionRecord(
            approval_id="appr_123",
            approved=True,
            decided_by="   ",
        )


def test_approval_request_record_creation() -> None:
    """Verify ApprovalRequestRecord instantiates correctly with valid timestamps."""
    now = utc_now()
    expires = now + timedelta(minutes=15)
    rec = ApprovalRequestRecord(
        workflow_id="wf_100",
        thread_id="th_100",
        step_id="step_1",
        tool_name="send_gmail_message",
        integration_name="gmail",
        description="Send email to user@example.com",
        arguments={"recipient": "user@example.com", "subject": "Hello"},
        arguments_hash="fakehash123",
        permission_level=PermissionLevel.SENSITIVE,
        created_at=now,
        expires_at=expires,
    )
    assert rec.approval_id.startswith("appr_")
    assert rec.status == ApprovalStatus.PENDING
    assert rec.request_version == 1
    assert rec.is_expired() is False


def test_approval_request_expiration_validation() -> None:
    """Verify ApprovalRequestRecord validates that expires_at is after created_at."""
    now = datetime.now(UTC)
    with pytest.raises(ApprovalError, match="must be after created_at"):
        ApprovalRequestRecord(
            workflow_id="wf_100",
            thread_id="th_100",
            step_id="step_1",
            tool_name="send_gmail_message",
            integration_name="gmail",
            description="Send email",
            arguments={},
            arguments_hash="fakehash123",
            permission_level=PermissionLevel.SENSITIVE,
            created_at=now,
            expires_at=now - timedelta(seconds=10),
        )


def test_approval_request_is_expired() -> None:
    """Verify is_expired correctly reflects the deadline."""
    now = utc_now()
    rec = ApprovalRequestRecord(
        workflow_id="wf_100",
        thread_id="th_100",
        step_id="step_1",
        tool_name="delete_calendar_event",
        integration_name="google_calendar",
        description="Delete event",
        arguments={},
        arguments_hash="hash",
        permission_level=PermissionLevel.SENSITIVE,
        created_at=now,
        expires_at=now + timedelta(seconds=2),
    )
    assert rec.is_expired(at_time=now) is False
    future = now + timedelta(seconds=5)
    assert rec.is_expired(at_time=future) is True


def test_api_models_serialization() -> None:
    """Verify Pydantic models for API request/response serialization."""
    req = ApprovalDecisionRequest(decided_by="bob_operator", reason="Approved by management")
    assert req.decided_by == "bob_operator"

    rec = ApprovalRequestRecord(
        workflow_id="wf_200",
        thread_id="th_200",
        step_id="step_2",
        tool_name="upload_drive_file",
        integration_name="google_drive",
        description="Upload test.txt",
        arguments={"name": "test.txt"},
        arguments_hash="hash_abc",
        permission_level=PermissionLevel.WRITE,
        expires_at=utc_now() + timedelta(minutes=10),
    )
    resp = ApprovalResponse(
        success=True,
        message="Approved successfully",
        approval=rec,
        decision_token="token_xyz",
    )
    dumped = resp.model_dump(mode="json")
    assert dumped["success"] is True
    assert dumped["approval"]["tool_name"] == "upload_drive_file"
    assert dumped["decision_token"] == "token_xyz"

    list_resp = ApprovalListResponse(total=1, items=[rec])
    assert list_resp.total == 1
    assert len(list_resp.items) == 1
