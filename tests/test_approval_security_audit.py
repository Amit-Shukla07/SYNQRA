"""Comprehensive Security Audit Tests for SYNQRA Approval Subsystem.

Covers:
1. Hashed token storage and constant-time authentication.
2. High entropy token validation in production.
3. Primary API key role isolation and explicit privilege configuration.
4. Operator listing and detail authorization scoping (no metadata leakage).
5. Graph resume context binding (permission tier, thread ID, tool name, arguments tamper).
6. Concurrency and single-use replay protection.
"""

import asyncio
import hashlib
from datetime import timedelta

import pytest
from starlette.testclient import TestClient

from synqra.agents.approval.auth import (
    ApprovalAuthenticator,
    OperatorIdentity,
    set_approval_authenticator,
)
from synqra.agents.approval.models import utc_now
from synqra.agents.approval.service import (
    ApprovalService,
    set_approval_service,
)
from synqra.agents.approval.store import InMemoryApprovalStore
from synqra.agents.schemas import ApprovalStatus, PermissionLevel
from synqra.server.mcp_server import create_mcp_server

RAW_SECRET_TOKEN_A: str = "token-secret-alpha-high-entropy-32-bytes"
RAW_SECRET_TOKEN_B: str = "token-secret-bravo-high-entropy-32-bytes"
RAW_ADMIN_TOKEN: str = "token-secret-admin-high-entropy-32-bytes"


# ---------------------------------------------------------------------------
# 1. Hashed Token Storage & Entropy Tests
# ---------------------------------------------------------------------------


def test_hashed_token_storage_and_constant_time_auth() -> None:
    """Verify that credentials can be stored as SHA-256 hashes and authenticate constant-time."""
    token_b_hash = hashlib.sha256(RAW_SECRET_TOKEN_B.encode("utf-8")).hexdigest()

    # Configure authenticator using pre-hashed token storage
    auth = ApprovalAuthenticator(
        token_mapping={
            f"sha256:{token_b_hash}": OperatorIdentity(
                operator_id="operator_bravo",
                roles=["operator"],
                allowed_workflows=["wf_bravo"],
            )
        },
        environment="development",
    )

    # 1. Raw secret token is NEVER stored in memory
    assert RAW_SECRET_TOKEN_B not in auth._token_hashes
    assert token_b_hash in auth._token_hashes

    # 2. Authenticating with raw bearer token succeeds and derives server-mapped identity
    is_auth, op, err = auth.authenticate_bearer_token(f"Bearer {RAW_SECRET_TOKEN_B}")
    assert is_auth is True
    assert err is None
    assert op is not None
    assert op.operator_id == "operator_bravo"

    # 3. Invalid token fails closed
    is_auth_inv, op_inv, err_inv = auth.authenticate_bearer_token(
        "Bearer wrong-token-12345678901234567890"
    )
    assert is_auth_inv is False
    assert op_inv is None
    assert "Invalid or unrecognized bearer token" in str(err_inv)


def test_production_entropy_enforcement() -> None:
    """Verify that in production, low-entropy tokens (< 32 chars) are strictly rejected."""
    short_token = "short-key-123"
    auth_prod = ApprovalAuthenticator(
        primary_api_key=short_token,
        environment="production",
    )

    # Short token is rejected in production
    is_auth, op, err = auth_prod.authenticate_bearer_token(f"Bearer {short_token}")
    assert is_auth is False
    assert op is None
    assert "production entropy requirements" in str(err)

    # High entropy token (>= 32 chars) is accepted
    long_token = "a" * 32
    auth_prod.register_token(long_token, OperatorIdentity(operator_id="long_op"))
    is_auth_long, op_long, err_long = auth_prod.authenticate_bearer_token(f"Bearer {long_token}")
    assert is_auth_long is True
    assert err_long is None
    assert op_long is not None


# ---------------------------------------------------------------------------
# 2. Primary API Key Role Isolation Tests
# ---------------------------------------------------------------------------


def test_primary_api_key_role_isolation() -> None:
    """Verify primary API key defaults strictly to 'operator' and cannot perform admin actions."""
    # 1. Default setup: no admin role
    default_auth = ApprovalAuthenticator(
        primary_api_key=RAW_SECRET_TOKEN_A,
        default_operator="operator_alpha",
    )
    is_auth, op, _ = default_auth.authenticate_bearer_token(f"Bearer {RAW_SECRET_TOKEN_A}")
    assert is_auth is True
    assert op is not None
    assert op.roles == ["operator"]
    assert "admin" not in op.roles

    # Cannot authorize for admin required role
    can_admin, err_admin = default_auth.authorize_operator(op, required_role="admin")
    assert can_admin is False
    assert "lacks required role 'admin'" in str(err_admin)

    # 2. Explicit admin configuration grants admin privileges
    admin_auth = ApprovalAuthenticator(
        primary_api_key=RAW_ADMIN_TOKEN,
        default_operator="admin_root",
        primary_roles=["operator", "admin"],
    )
    is_auth_adm, op_adm, _ = admin_auth.authenticate_bearer_token(f"Bearer {RAW_ADMIN_TOKEN}")
    assert is_auth_adm is True
    assert op_adm is not None
    assert "admin" in op_adm.roles

    can_admin_ok, _ = admin_auth.authorize_operator(op_adm, required_role="admin")
    assert can_admin_ok is True


def test_admin_restricted_workflows_cannot_exceed_scope() -> None:
    """Verify that an administrator with explicit workflow scoping cannot exceed its scope."""
    auth = ApprovalAuthenticator()
    scoped_admin = OperatorIdentity(
        operator_id="scoped_admin",
        roles=["operator", "admin"],
        allowed_workflows=["wf_dept_a"],
    )

    # Allowed inside designated workflow
    ok_in, err_in = auth.authorize_operator(scoped_admin, workflow_id="wf_dept_a")
    assert ok_in is True
    assert err_in is None

    # Blocked outside designated workflow, even though user has 'admin' role!
    ok_out, err_out = auth.authorize_operator(scoped_admin, workflow_id="wf_dept_b")
    assert ok_out is False
    assert "not authorized to manage workflow" in str(err_out)


# ---------------------------------------------------------------------------
# 3. Listing & Detail Scoping API Tests (No Metadata Leakage)
# ---------------------------------------------------------------------------


@pytest.fixture
def audit_app_client() -> tuple[TestClient, ApprovalService]:
    """Test client configured with two isolated operators."""
    service = ApprovalService(store=InMemoryApprovalStore())
    set_approval_service(service)

    auth = ApprovalAuthenticator(
        token_mapping={
            RAW_SECRET_TOKEN_A: OperatorIdentity(
                operator_id="alice",
                roles=["operator"],
                allowed_workflows=["wf_alice_only"],
            ),
            RAW_SECRET_TOKEN_B: OperatorIdentity(
                operator_id="bob",
                roles=["operator"],
                allowed_workflows=["wf_bob_only"],
            ),
        },
        environment="testing",
    )
    set_approval_authenticator(auth)

    server = create_mcp_server()
    client = TestClient(server.http_app())
    return client, service


@pytest.mark.asyncio
async def test_operator_listing_scoping_and_cross_workflow_blocking(
    audit_app_client: tuple[TestClient, ApprovalService],
) -> None:
    """Verify operator A cannot list or probe operator B's workflows."""
    client, service = audit_app_client

    # Create approval for Alice's workflow
    await service.create_approval_request(
        workflow_id="wf_alice_only",
        thread_id="th_alice",
        step_id="step_a",
        tool_name="upload_drive_file",
        arguments={"name": "alice_secrets.pdf"},
        permission_level=PermissionLevel.WRITE,
    )

    # Create approval for Bob's workflow
    await service.create_approval_request(
        workflow_id="wf_bob_only",
        thread_id="th_bob",
        step_id="step_b",
        tool_name="send_gmail_message",
        arguments={"recipient": "bob_client@corp.com"},
        permission_level=PermissionLevel.SENSITIVE,
    )

    # Alice lists all approvals: should only see wf_alice_only
    res_alice = client.get(
        "/api/approvals",
        headers={"Authorization": f"Bearer {RAW_SECRET_TOKEN_A}"},
    )
    assert res_alice.status_code == 200
    data_alice = res_alice.json()
    assert data_alice["total"] == 1
    assert data_alice["items"][0]["workflow_id"] == "wf_alice_only"

    # Alice attempts to probe Bob's workflow explicitly -> 403 Forbidden
    res_probe = client.get(
        "/api/approvals?workflow_id=wf_bob_only",
        headers={"Authorization": f"Bearer {RAW_SECRET_TOKEN_A}"},
    )
    assert res_probe.status_code == 403
    assert "not authorized for requested workflow" in res_probe.json()["detail"]


@pytest.mark.asyncio
async def test_operator_detail_and_decision_isolation_no_metadata_leakage(
    audit_app_client: tuple[TestClient, ApprovalService],
) -> None:
    """Verify operator A cannot view or decide operator B's approval (no metadata leaked)."""
    client, service = audit_app_client

    # Create private approval for Bob
    rec_b = await service.create_approval_request(
        workflow_id="wf_bob_only",
        thread_id="th_bob",
        step_id="step_b",
        tool_name="delete_calendar_event",
        arguments={"event_id": "ev_confidential_999"},
        permission_level=PermissionLevel.WRITE,
    )

    # 1. Alice attempts to inspect Bob's approval details -> 403 Forbidden without leaking metadata
    res_get = client.get(
        f"/api/approvals/{rec_b.approval_id}",
        headers={"Authorization": f"Bearer {RAW_SECRET_TOKEN_A}"},
    )
    assert res_get.status_code == 403
    # Crucial security assertion: private details and workflow ID are not disclosed
    assert "wf_bob_only" not in res_get.text
    assert "ev_confidential_999" not in res_get.text
    assert "insufficient permissions" in res_get.json()["detail"]

    # 2. Alice attempts to approve Bob's approval -> 403 Forbidden
    res_approve = client.post(
        f"/api/approvals/{rec_b.approval_id}/approve",
        headers={"Authorization": f"Bearer {RAW_SECRET_TOKEN_A}"},
        json={"reason": "Attacker trying to approve Bob's action"},
    )
    assert res_approve.status_code == 403
    assert "wf_bob_only" not in res_approve.text

    # 3. Alice attempts to reject Bob's approval -> 403 Forbidden
    res_reject = client.post(
        f"/api/approvals/{rec_b.approval_id}/reject",
        headers={"Authorization": f"Bearer {RAW_SECRET_TOKEN_A}"},
        json={"reason": "Attacker trying to reject Bob's action"},
    )
    assert res_reject.status_code == 403
    assert "wf_bob_only" not in res_reject.text


# ---------------------------------------------------------------------------
# 4. Graph Resume Context Binding & Tamper Resistance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_context_binding_strict_verifications() -> None:
    """Verify consumption verifies thread, tool, permission level, and exact arguments hash."""
    service = ApprovalService(store=InMemoryApprovalStore())
    args = {"query": "quarterly_results.xlsx"}

    rec = await service.create_approval_request(
        workflow_id="wf_sec",
        thread_id="th_sec_1",
        step_id="step_sec",
        tool_name="download_drive_file",
        arguments=args,
        permission_level=PermissionLevel.WRITE,
    )
    await service.approve_request(rec.approval_id, decided_by="lead_op")

    # 1. Thread ID mismatch rejected
    ok_th, err_th = await service.consume_approval(
        approval_id=rec.approval_id,
        workflow_id="wf_sec",
        thread_id="th_ATTACKER_THREAD",
        step_id="step_sec",
        tool_name="download_drive_file",
        arguments=args,
        permission_level=PermissionLevel.WRITE,
    )
    assert ok_th is False
    assert "Thread mismatch" in str(err_th)

    # 2. Tool name mismatch rejected
    ok_tool, err_tool = await service.consume_approval(
        approval_id=rec.approval_id,
        workflow_id="wf_sec",
        thread_id="th_sec_1",
        step_id="step_sec",
        tool_name="delete_calendar_event",
        arguments=args,
        permission_level=PermissionLevel.WRITE,
    )
    assert ok_tool is False
    assert "Tool mismatch" in str(err_tool)

    # 3. Permission level mismatch rejected
    ok_perm, err_perm = await service.consume_approval(
        approval_id=rec.approval_id,
        workflow_id="wf_sec",
        thread_id="th_sec_1",
        step_id="step_sec",
        tool_name="download_drive_file",
        arguments=args,
        permission_level=PermissionLevel.SENSITIVE,
    )
    assert ok_perm is False
    assert "Permission level mismatch" in str(err_perm)

    # 4. Tampered arguments rejected
    ok_arg, err_arg = await service.consume_approval(
        approval_id=rec.approval_id,
        workflow_id="wf_sec",
        thread_id="th_sec_1",
        step_id="step_sec",
        tool_name="download_drive_file",
        arguments={"query": "passwords.txt"},
        permission_level=PermissionLevel.WRITE,
    )
    assert ok_arg is False
    assert "Argument tamper detected" in str(err_arg)

    # 5. Legitimate context succeeds
    ok_valid, err_valid = await service.consume_approval(
        approval_id=rec.approval_id,
        workflow_id="wf_sec",
        thread_id="th_sec_1",
        step_id="step_sec",
        tool_name="download_drive_file",
        arguments=args,
        permission_level=PermissionLevel.WRITE,
    )
    assert ok_valid is True
    assert err_valid is None


# ---------------------------------------------------------------------------
# 5. Concurrency and Replay Protection Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_replay_after_rejection_blocked() -> None:
    """Verify that an approval rejected by an operator cannot be consumed."""
    service = ApprovalService(store=InMemoryApprovalStore())
    args = {"query": "confidential.pdf"}

    rec = await service.create_approval_request(
        workflow_id="wf_rej",
        thread_id="th_rej",
        step_id="step_rej",
        tool_name="download_drive_file",
        arguments=args,
        permission_level=PermissionLevel.WRITE,
    )
    await service.reject_request(rec.approval_id, decided_by="security_officer")

    # Attempt to consume rejected approval
    ok, err = await service.consume_approval(
        approval_id=rec.approval_id,
        workflow_id="wf_rej",
        thread_id="th_rej",
        step_id="step_rej",
        tool_name="download_drive_file",
        arguments=args,
        permission_level=PermissionLevel.WRITE,
    )
    assert ok is False
    assert "was rejected" in str(err)


@pytest.mark.asyncio
async def test_expiration_during_resume_race_blocked() -> None:
    """Verify that if an approval expires before consumption, consumption fails."""
    service = ApprovalService(store=InMemoryApprovalStore())
    args = {"query": "timely.pdf"}

    rec = await service.create_approval_request(
        workflow_id="wf_race",
        thread_id="th_race",
        step_id="step_race",
        tool_name="download_drive_file",
        arguments=args,
        permission_level=PermissionLevel.WRITE,
        validity_seconds=30,
    )
    await service.approve_request(rec.approval_id, decided_by="admin_alice")

    # Simulate expiration by backdating record in store
    expired_rec = rec.model_copy(
        update={
            "status": ApprovalStatus.APPROVED,
            "expires_at": utc_now() - timedelta(seconds=5),
        }
    )
    await service.store.save(expired_rec)

    # Attempting to consume an expired approval fails
    ok, err = await service.consume_approval(
        approval_id=rec.approval_id,
        workflow_id="wf_race",
        thread_id="th_race",
        step_id="step_race",
        tool_name="download_drive_file",
        arguments=args,
        permission_level=PermissionLevel.WRITE,
    )
    assert ok is False
    assert "has expired" in str(err)


@pytest.mark.asyncio
async def test_concurrent_consumption_race_exactly_one_executes() -> None:
    """Verify race condition: two simultaneous consumption calls execute exactly once."""
    service = ApprovalService(store=InMemoryApprovalStore())
    args = {"recipient": "client@example.com", "subject": "Financial Report"}

    rec = await service.create_approval_request(
        workflow_id="wf_concurrent_consume",
        thread_id="th_concurrent",
        step_id="step_email",
        tool_name="send_gmail_message",
        arguments=args,
        permission_level=PermissionLevel.SENSITIVE,
    )
    await service.approve_request(rec.approval_id, decided_by="lead_exec")

    # Launch two simultaneous coroutines attempting to consume
    res1, res2 = await asyncio.gather(
        service.consume_approval(
            approval_id=rec.approval_id,
            workflow_id="wf_concurrent_consume",
            thread_id="th_concurrent",
            step_id="step_email",
            tool_name="send_gmail_message",
            arguments=args,
            permission_level=PermissionLevel.SENSITIVE,
        ),
        service.consume_approval(
            approval_id=rec.approval_id,
            workflow_id="wf_concurrent_consume",
            thread_id="th_concurrent",
            step_id="step_email",
            tool_name="send_gmail_message",
            arguments=args,
            permission_level=PermissionLevel.SENSITIVE,
        ),
    )

    # Exactly one succeeded and one failed
    results = [res1[0], res2[0]]
    assert sorted(results) == [False, True]

    # Verify failure reason on the loser
    failed_err = res1[1] if not res1[0] else res2[1]
    assert "consumed" in str(failed_err).lower()
