"""Unit tests for the ApprovalService business logic and concurrency safety."""

import asyncio
from datetime import timedelta

import pytest

from synqra.agents.approval.models import utc_now
from synqra.agents.approval.service import ApprovalService
from synqra.agents.approval.store import InMemoryApprovalStore
from synqra.agents.exceptions import ApprovalError
from synqra.agents.schemas import ApprovalStatus, PermissionLevel


@pytest.fixture
def approval_service() -> ApprovalService:
    """Fixture providing a fresh isolated ApprovalService instance."""
    return ApprovalService(store=InMemoryApprovalStore())


@pytest.mark.asyncio
async def test_create_and_get_approval_request(approval_service: ApprovalService) -> None:
    """Verify creating and retrieving an approval request record."""
    rec = await approval_service.create_approval_request(
        workflow_id="wf_001",
        thread_id="th_001",
        step_id="step_1",
        tool_name="send_gmail_message",
        arguments={"recipient": "ceo@example.com", "subject": "Quarterly Report", "body": "Report"},
        permission_level=PermissionLevel.SENSITIVE,
        validity_seconds=600,
    )

    assert rec.approval_id.startswith("appr_")
    assert rec.workflow_id == "wf_001"
    assert rec.thread_id == "th_001"
    assert rec.step_id == "step_1"
    assert rec.tool_name == "send_gmail_message"
    assert rec.integration_name == "gmail"
    assert rec.status == ApprovalStatus.PENDING
    assert "ceo@example.com" in rec.description
    assert rec.arguments_hash is not None

    fetched = await approval_service.get_approval_request(rec.approval_id)
    assert fetched is not None
    assert fetched.approval_id == rec.approval_id
    assert fetched.status == ApprovalStatus.PENDING


@pytest.mark.asyncio
async def test_list_pending_approvals(approval_service: ApprovalService) -> None:
    """Verify listing pending requests and filtering by workflow ID."""
    await approval_service.create_approval_request(
        workflow_id="wf_A",
        thread_id="th_A",
        step_id="step_1",
        tool_name="create_calendar_event",
        arguments={"summary": "Sync"},
        permission_level=PermissionLevel.WRITE,
    )
    await approval_service.create_approval_request(
        workflow_id="wf_B",
        thread_id="th_B",
        step_id="step_2",
        tool_name="upload_drive_file",
        arguments={"name": "notes.md"},
        permission_level=PermissionLevel.WRITE,
    )

    all_pending = await approval_service.list_pending_approvals()
    assert len(all_pending) == 2

    wf_a_pending = await approval_service.list_pending_approvals(workflow_id="wf_A")
    assert len(wf_a_pending) == 1
    assert wf_a_pending[0].workflow_id == "wf_A"


@pytest.mark.asyncio
async def test_approve_request_success(approval_service: ApprovalService) -> None:
    """Verify approving a request transitions status and issues an HMAC binding token."""
    rec = await approval_service.create_approval_request(
        workflow_id="wf_101",
        thread_id="th_101",
        step_id="step_1",
        tool_name="delete_calendar_event",
        arguments={"calendar_id": "primary", "event_id": "ev_999"},
        permission_level=PermissionLevel.SENSITIVE,
    )

    updated, token = await approval_service.approve_request(
        approval_id=rec.approval_id,
        decided_by="admin_operator",
        reason="Approved after user confirmation.",
    )

    assert updated.status == ApprovalStatus.APPROVED
    assert updated.binding_token == token
    assert token is not None
    assert updated.decision is not None
    assert updated.decision.approved is True
    assert updated.decision.decided_by == "admin_operator"
    assert updated.decision.reason == "Approved after user confirmation."


@pytest.mark.asyncio
async def test_reject_request_success(approval_service: ApprovalService) -> None:
    """Verify rejecting a request transitions status and blocks authorization."""
    rec = await approval_service.create_approval_request(
        workflow_id="wf_102",
        thread_id="th_102",
        step_id="step_1",
        tool_name="create_google_document",
        arguments={"title": "Draft Plan"},
        permission_level=PermissionLevel.WRITE,
    )

    updated = await approval_service.reject_request(
        approval_id=rec.approval_id,
        decided_by="reviewer_carol",
        reason="Content requires revisions.",
    )

    assert updated.status == ApprovalStatus.REJECTED
    assert updated.decision is not None
    assert updated.decision.approved is False
    assert updated.decision.decided_by == "reviewer_carol"


@pytest.mark.asyncio
async def test_duplicate_decision_prevention(approval_service: ApprovalService) -> None:
    """Verify that a request cannot be approved or rejected more than once."""
    rec = await approval_service.create_approval_request(
        workflow_id="wf_103",
        thread_id="th_103",
        step_id="step_1",
        tool_name="upload_drive_file",
        arguments={"name": "file.pdf"},
        permission_level=PermissionLevel.WRITE,
    )

    # First approval succeeds
    await approval_service.approve_request(
        approval_id=rec.approval_id,
        decided_by="admin_user",
    )

    # Second approval attempt fails with ApprovalError
    with pytest.raises(ApprovalError, match="cannot be approved: status is approved"):
        await approval_service.approve_request(
            approval_id=rec.approval_id,
            decided_by="admin_user_2",
        )

    # Rejection attempt on approved request fails
    with pytest.raises(ApprovalError, match="cannot be rejected: status is approved"):
        await approval_service.reject_request(
            approval_id=rec.approval_id,
            decided_by="admin_user_2",
        )


@pytest.mark.asyncio
async def test_expiration_handling(approval_service: ApprovalService) -> None:
    """Verify that expired requests cannot be approved and transition to EXPIRED."""
    # Create request with 30s validity
    rec = await approval_service.create_approval_request(
        workflow_id="wf_104",
        thread_id="th_104",
        step_id="step_1",
        tool_name="send_gmail_message",
        arguments={"recipient": "test@test.com", "subject": "Hi"},
        permission_level=PermissionLevel.SENSITIVE,
        validity_seconds=30,
    )

    # Manually backdate expiration to simulate time passing
    past = utc_now() - timedelta(seconds=10)
    await approval_service.store.save(rec.model_copy(update={"expires_at": past}))

    # Pending list should not include expired
    pending = await approval_service.list_pending_approvals()
    assert rec.approval_id not in [p.approval_id for p in pending]

    # Attempting to approve an expired request raises ApprovalError
    with pytest.raises(ApprovalError, match="expired"):
        await approval_service.approve_request(
            approval_id=rec.approval_id,
            decided_by="alice",
        )


@pytest.mark.asyncio
async def test_single_use_consumption(approval_service: ApprovalService) -> None:
    """Verify single-use consumption: first consume succeeds, second call fails."""
    args = {"recipient": "boss@corp.com", "subject": "Urgent"}
    rec = await approval_service.create_approval_request(
        workflow_id="wf_105",
        thread_id="th_105",
        step_id="step_send",
        tool_name="send_gmail_message",
        arguments=args,
        permission_level=PermissionLevel.SENSITIVE,
    )

    await approval_service.approve_request(
        approval_id=rec.approval_id,
        decided_by="human_approver",
    )

    # 1. First consumption succeeds
    success, err = await approval_service.consume_approval(
        approval_id=rec.approval_id,
        workflow_id="wf_105",
        thread_id="th_105",
        step_id="step_send",
        tool_name="send_gmail_message",
        arguments=args,
    )
    assert success is True
    assert err is None

    # Check status is now CONSUMED
    stored = await approval_service.get_approval_request(rec.approval_id)
    assert stored is not None
    assert stored.status == ApprovalStatus.CONSUMED
    assert stored.consumed_by == "wf_105:step_send"
    assert stored.consumed_at is not None

    # 2. Replay attempt fails: cannot consume twice!
    success_retry, err_retry = await approval_service.consume_approval(
        approval_id=rec.approval_id,
        workflow_id="wf_105",
        thread_id="th_105",
        step_id="step_send",
        tool_name="send_gmail_message",
        arguments=args,
    )
    assert success_retry is False
    assert "already been consumed" in str(err_retry)


@pytest.mark.asyncio
async def test_concurrent_approval_attempts(approval_service: ApprovalService) -> None:
    """Verify that two concurrent approval calls cannot both succeed."""
    rec = await approval_service.create_approval_request(
        workflow_id="wf_concurrent",
        thread_id="th_concurrent",
        step_id="step_1",
        tool_name="upload_drive_file",
        arguments={"name": "concurrent.txt"},
        permission_level=PermissionLevel.WRITE,
    )

    async def _try_approve(actor: str) -> bool:
        try:
            await approval_service.approve_request(
                approval_id=rec.approval_id,
                decided_by=actor,
            )
            return True
        except ApprovalError:
            return False

    results = await asyncio.gather(
        _try_approve("admin_1"),
        _try_approve("admin_2"),
    )

    # Exactly one must succeed, one must fail
    assert sorted(results) == [False, True]


@pytest.mark.asyncio
async def test_concurrent_consumption_attempts(approval_service: ApprovalService) -> None:
    """Verify that two concurrent consume calls cannot both consume the approval."""
    args = {"name": "unique.txt"}
    rec = await approval_service.create_approval_request(
        workflow_id="wf_consume_race",
        thread_id="th_consume_race",
        step_id="step_1",
        tool_name="upload_drive_file",
        arguments=args,
        permission_level=PermissionLevel.WRITE,
    )

    await approval_service.approve_request(
        approval_id=rec.approval_id,
        decided_by="admin_user",
    )

    async def _try_consume() -> bool:
        succ, _ = await approval_service.consume_approval(
            approval_id=rec.approval_id,
            workflow_id="wf_consume_race",
            thread_id="th_consume_race",
            step_id="step_1",
            tool_name="upload_drive_file",
            arguments=args,
        )
        return succ

    results = await asyncio.gather(_try_consume(), _try_consume())
    assert sorted(results) == [False, True]
