"""Unit tests for the Approval Security subsystem.

Verifies canonical serialization, SHA-256 fingerprinting, HMAC signature verification,
strict context binding, replay protection, and human identity verification.
"""

from datetime import timedelta

from synqra.agents.approval.models import ApprovalRequestRecord, utc_now
from synqra.agents.approval.security import (
    canonical_json_serialize,
    compute_arguments_hash,
    format_human_readable_operation,
    generate_approval_binding_token,
    validate_human_actor,
    verify_approval_binding,
)
from synqra.agents.schemas import ApprovalStatus, PermissionLevel


def test_canonical_json_serialization() -> None:
    """Verify deterministic serialization regardless of key ordering."""
    dict_a = {"b": 2, "a": 1, "c": {"z": 10, "y": 20}}
    dict_b = {"a": 1, "c": {"y": 20, "z": 10}, "b": 2}

    json_a = canonical_json_serialize(dict_a)
    json_b = canonical_json_serialize(dict_b)

    assert json_a == json_b
    assert json_a == '{"a":1,"b":2,"c":{"y":20,"z":10}}'


def test_arguments_hash_tamper_detection() -> None:
    """Verify compute_arguments_hash detects any argument modifications."""
    args_1 = {"recipient": "user@example.com", "subject": "Hello"}
    args_2 = {"subject": "Hello", "recipient": "user@example.com"}
    args_tampered = {"recipient": "attacker@evil.com", "subject": "Hello"}

    hash_1 = compute_arguments_hash(args_1)
    hash_2 = compute_arguments_hash(args_2)
    hash_tampered = compute_arguments_hash(args_tampered)

    assert hash_1 == hash_2
    assert hash_1 != hash_tampered


def test_binding_token_generation_and_tamper_detection() -> None:
    """Verify HMAC signature generation and detection of forged tokens."""
    token = generate_approval_binding_token(
        approval_id="appr_01",
        workflow_id="wf_01",
        thread_id="th_01",
        step_id="step_1",
        tool_name="send_gmail_message",
        arguments_hash="hash_123",
        permission_level=PermissionLevel.SENSITIVE,
        request_version=1,
        secret_key="my-secret-key",
    )
    assert isinstance(token, str)
    assert len(token) == 64

    # Token with different secret or arguments hash will not match
    different_token = generate_approval_binding_token(
        approval_id="appr_01",
        workflow_id="wf_01",
        thread_id="th_01",
        step_id="step_1",
        tool_name="send_gmail_message",
        arguments_hash="tampered_hash",
        permission_level=PermissionLevel.SENSITIVE,
        request_version=1,
        secret_key="my-secret-key",
    )
    assert token != different_token


def test_validate_human_actor() -> None:
    """Verify validate_human_actor accepts human identifiers and rejects LLMs/bots."""
    valid, err = validate_human_actor("alice_operator")
    assert valid is True
    assert err is None

    empty, err = validate_human_actor("  ")
    assert empty is False
    assert "empty" in str(err)

    none_actor, err = validate_human_actor(None)
    assert none_actor is False
    assert "empty" in str(err)

    for disallowed in ["gpt-4o", "llm", "ai", "assistant", "system", "claude"]:
        ok, err = validate_human_actor(disallowed)
        assert ok is False
        assert "Automated models" in str(err)


def test_verify_approval_binding_all_checks() -> None:
    """Verify verify_approval_binding enforces exact match on all binding dimensions."""
    now = utc_now()
    args = {"recipient": "alice@synqra.ai", "subject": "Test"}
    arg_hash = compute_arguments_hash(args)
    secret = "test-secret"

    token = generate_approval_binding_token(
        approval_id="appr_bind_1",
        workflow_id="wf_bind_1",
        thread_id="th_bind_1",
        step_id="step_bind_1",
        tool_name="send_gmail_message",
        arguments_hash=arg_hash,
        permission_level=PermissionLevel.SENSITIVE,
        request_version=1,
        secret_key=secret,
    )

    record = ApprovalRequestRecord(
        approval_id="appr_bind_1",
        workflow_id="wf_bind_1",
        thread_id="th_bind_1",
        step_id="step_bind_1",
        tool_name="send_gmail_message",
        integration_name="gmail",
        description="Send email",
        arguments=args,
        arguments_hash=arg_hash,
        permission_level=PermissionLevel.SENSITIVE,
        status=ApprovalStatus.APPROVED,
        binding_token=token,
        created_at=now,
        expires_at=now + timedelta(minutes=10),
    )

    # 1. Valid verification
    valid, err = verify_approval_binding(
        record=record,
        workflow_id="wf_bind_1",
        thread_id="th_bind_1",
        step_id="step_bind_1",
        tool_name="send_gmail_message",
        candidate_arguments=args,
        secret_key=secret,
    )
    assert valid is True
    assert err is None

    # 2. Modified arguments (tamper)
    valid, err = verify_approval_binding(
        record=record,
        workflow_id="wf_bind_1",
        thread_id="th_bind_1",
        step_id="step_bind_1",
        tool_name="send_gmail_message",
        candidate_arguments={"recipient": "attacker@evil.com", "subject": "Test"},
        secret_key=secret,
    )
    assert valid is False
    assert "tamper detected" in str(err)

    # 3. Wrong tool name
    valid, err = verify_approval_binding(
        record=record,
        workflow_id="wf_bind_1",
        thread_id="th_bind_1",
        step_id="step_bind_1",
        tool_name="delete_calendar_event",
        candidate_arguments=args,
        secret_key=secret,
    )
    assert valid is False
    assert "Tool mismatch" in str(err)

    # 4. Wrong workflow ID
    valid, err = verify_approval_binding(
        record=record,
        workflow_id="wf_DIFFERENT",
        thread_id="th_bind_1",
        step_id="step_bind_1",
        tool_name="send_gmail_message",
        candidate_arguments=args,
        secret_key=secret,
    )
    assert valid is False
    assert "Workflow mismatch" in str(err)

    # 5. Wrong step ID
    valid, err = verify_approval_binding(
        record=record,
        workflow_id="wf_bind_1",
        thread_id="th_bind_1",
        step_id="step_DIFFERENT",
        tool_name="send_gmail_message",
        candidate_arguments=args,
        secret_key=secret,
    )
    assert valid is False
    assert "Step mismatch" in str(err)

    # 6. Status is not APPROVED (e.g. CONSUMED or REJECTED)
    consumed_rec = record.model_copy(update={"status": ApprovalStatus.CONSUMED})
    valid, err = verify_approval_binding(
        record=consumed_rec,
        workflow_id="wf_bind_1",
        thread_id="th_bind_1",
        step_id="step_bind_1",
        tool_name="send_gmail_message",
        candidate_arguments=args,
        secret_key=secret,
    )
    assert valid is False
    assert "already been consumed" in str(err)

    # 7. Expired record
    valid, err = verify_approval_binding(
        record=record,
        workflow_id="wf_bind_1",
        thread_id="th_bind_1",
        step_id="step_bind_1",
        tool_name="send_gmail_message",
        candidate_arguments=args,
        secret_key=secret,
        at_time=now + timedelta(minutes=20),
    )
    assert valid is False
    assert "expired" in str(err)


def test_format_human_readable_operation() -> None:
    """Verify clear human-readable operations are rendered for all sensitive tools."""
    gmail_desc = format_human_readable_operation(
        "send_gmail_message",
        {"recipient": "boss@corp.com", "subject": "Hello", "body": "Weekly review notes"},
    )
    assert "boss@corp.com" in gmail_desc
    assert "Weekly review notes" in gmail_desc

    cal_desc = format_human_readable_operation(
        "delete_calendar_event",
        {"calendar_id": "primary", "event_id": "ev_456"},
    )
    assert "DELETE calendar event 'ev_456'" in cal_desc

    drive_desc = format_human_readable_operation(
        "create_google_document",
        {"title": "Product Roadmap 2027"},
    )
    assert "Product Roadmap 2027" in drive_desc
