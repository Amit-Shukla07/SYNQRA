"""Integration tests for Human-in-the-Loop Approval REST API authentication and authorization."""

import pytest
from starlette.testclient import TestClient

from synqra.agents.approval.auth import (
    ApprovalAuthenticator,
    OperatorIdentity,
    set_approval_authenticator,
)
from synqra.agents.approval.service import (
    ApprovalService,
    set_approval_service,
)
from synqra.agents.approval.store import InMemoryApprovalStore
from synqra.agents.schemas import PermissionLevel
from synqra.server.mcp_server import create_mcp_server

TEST_ADMIN_TOKEN: str = "synqra-test-admin-secret-token-32bytes-long"
TEST_OPERATOR_TOKEN: str = "synqra-test-operator-bob-token-32bytes-long"


@pytest.fixture
def test_approval_service() -> ApprovalService:
    """Fixture providing a clean isolated ApprovalService singleton."""
    service = ApprovalService(store=InMemoryApprovalStore())
    set_approval_service(service)
    yield service
    set_approval_service(None)


@pytest.fixture
def test_authenticator() -> ApprovalAuthenticator:
    """Fixture configuring test authenticator with admin and scoped operator identities."""
    auth = ApprovalAuthenticator(
        primary_api_key=TEST_ADMIN_TOKEN,
        default_operator="admin_alice",
        primary_roles=["operator", "admin"],
        token_mapping={
            TEST_OPERATOR_TOKEN: OperatorIdentity(
                operator_id="operator_bob",
                roles=["operator"],
                allowed_workflows=["wf_allowed_bob"],
            )
        },
        environment="testing",
    )
    set_approval_authenticator(auth)
    yield auth
    set_approval_authenticator(None)


@pytest.fixture
def client(
    test_approval_service: ApprovalService, test_authenticator: ApprovalAuthenticator
) -> TestClient:
    """Fixture providing TestClient with active authenticator and approval service."""
    _ = (test_approval_service, test_authenticator)
    server = create_mcp_server()
    app = server.http_app()
    return TestClient(app)


def test_missing_bearer_token_rejected(client: TestClient) -> None:
    """Verify that requests without Authorization header are rejected with 401 Unauthorized."""
    res = client.get("/api/approvals")
    assert res.status_code == 401
    assert "Missing Authorization header" in res.json()["detail"]


def test_invalid_bearer_token_rejected(client: TestClient) -> None:
    """Verify that requests with incorrect/invalid bearer token are rejected with 401."""
    res = client.get(
        "/api/approvals",
        headers={"Authorization": "Bearer wrong-invalid-token-12345"},
    )
    assert res.status_code == 401
    assert "Invalid or unrecognized bearer token" in res.json()["detail"]


def test_malformed_authorization_header(client: TestClient) -> None:
    """Verify that Authorization headers not matching 'Bearer <token>' are rejected with 401."""
    res = client.get(
        "/api/approvals",
        headers={"Authorization": "Basic dXNlcjpwYXNz"},
    )
    assert res.status_code == 401
    assert "Malformed Authorization header" in res.json()["detail"]


def test_valid_bearer_token_lists_pending(client: TestClient) -> None:
    """Verify that authenticated requests with valid bearer token succeed."""
    res = client.get(
        "/api/approvals",
        headers={"Authorization": f"Bearer {TEST_ADMIN_TOKEN}"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["total"] == 0
    assert data["items"] == []


@pytest.mark.asyncio
async def test_impersonation_attempt_rejected(
    client: TestClient, test_approval_service: ApprovalService
) -> None:
    """Verify that attempting to assert a different operator identity is rejected with 403."""
    rec = await test_approval_service.create_approval_request(
        workflow_id="wf_imp",
        thread_id="th_imp",
        step_id="step_imp",
        tool_name="upload_drive_file",
        arguments={"name": "safe.txt"},
        permission_level=PermissionLevel.WRITE,
    )

    # Caller presents Alice's token, but attempts to self-assert actor 'attacker_eve'
    res = client.post(
        f"/api/approvals/{rec.approval_id}/approve",
        headers={
            "Authorization": f"Bearer {TEST_ADMIN_TOKEN}",
            "X-Synqra-Actor": "attacker_eve",
        },
        json={"reason": "Impersonation attempt"},
    )
    assert res.status_code == 403
    assert "Impersonation detected" in res.json()["detail"]


@pytest.mark.asyncio
async def test_workflow_authorization_boundaries(
    client: TestClient, test_approval_service: ApprovalService
) -> None:
    """Verify an operator cannot view or approve a workflow outside their allowed scope."""
    # Create request belonging to workflow 'wf_private_alice'
    rec = await test_approval_service.create_approval_request(
        workflow_id="wf_private_alice",
        thread_id="th_priv",
        step_id="step_priv",
        tool_name="upload_drive_file",
        arguments={"name": "private.txt"},
        permission_level=PermissionLevel.WRITE,
    )

    # Bob's token is only authorized for 'wf_allowed_bob'
    bob_headers = {"Authorization": f"Bearer {TEST_OPERATOR_TOKEN}"}

    # 1. Bob attempts GET on Alice's approval -> 403 Forbidden (no workflow metadata leaked)
    get_res = client.get(f"/api/approvals/{rec.approval_id}", headers=bob_headers)
    assert get_res.status_code == 403
    assert "insufficient permissions" in get_res.json()["detail"]

    # 2. Bob attempts POST approve on Alice's approval -> 403 Forbidden
    approve_res = client.post(
        f"/api/approvals/{rec.approval_id}/approve",
        headers=bob_headers,
        json={"reason": "Unauthorized approval"},
    )
    assert approve_res.status_code == 403
    assert "insufficient permissions" in approve_res.json()["detail"]


@pytest.mark.asyncio
async def test_successful_approval_derives_operator_identity(
    client: TestClient, test_approval_service: ApprovalService
) -> None:
    """Verify that operator identity is derived strictly from server-authenticated token."""
    rec = await test_approval_service.create_approval_request(
        workflow_id="wf_allowed_bob",
        thread_id="th_bob",
        step_id="step_bob",
        tool_name="upload_drive_file",
        arguments={"name": "bob_doc.txt"},
        permission_level=PermissionLevel.WRITE,
    )

    # Bob approves with valid credentials
    res = client.post(
        f"/api/approvals/{rec.approval_id}/approve",
        headers={"Authorization": f"Bearer {TEST_OPERATOR_TOKEN}"},
        json={"reason": "Authorized by Bob"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["approval"]["status"] == "approved"
    # Derived from server mapping, not caller header!
    assert data["approval"]["decision"]["decided_by"] == "operator_bob"


@pytest.mark.asyncio
async def test_successful_rejection_derives_operator_identity(
    client: TestClient, test_approval_service: ApprovalService
) -> None:
    """Verify that operator rejection identity is derived strictly from token mapping."""
    rec = await test_approval_service.create_approval_request(
        workflow_id="wf_allowed_bob",
        thread_id="th_bob",
        step_id="step_bob_2",
        tool_name="create_google_document",
        arguments={"title": "Proposal"},
        permission_level=PermissionLevel.WRITE,
    )

    res = client.post(
        f"/api/approvals/{rec.approval_id}/reject",
        headers={"Authorization": f"Bearer {TEST_OPERATOR_TOKEN}"},
        json={"reason": "Rejected by Bob"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["approval"]["status"] == "rejected"
    assert data["approval"]["decision"]["decided_by"] == "operator_bob"
