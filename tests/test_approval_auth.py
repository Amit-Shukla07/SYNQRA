"""Unit tests for the ApprovalAuthenticator and operator identity validation."""

from synqra.agents.approval.auth import (
    ApprovalAuthenticator,
    OperatorIdentity,
)


def test_authenticator_fails_closed_when_no_tokens_configured() -> None:
    """Verify that an unconfigured authenticator fails closed."""
    auth = ApprovalAuthenticator(primary_api_key=None, token_mapping=None)
    is_auth, op, err = auth.authenticate_bearer_token("Bearer some-token")
    assert is_auth is False
    assert op is None
    assert "not configured on the server" in str(err)


def test_authenticator_valid_token_derives_identity() -> None:
    """Verify constant-time token match derives server-mapped operator identity."""
    auth = ApprovalAuthenticator(
        primary_api_key="secret-master-token-12345",
        default_operator="root_admin",
        primary_roles=["operator", "admin"],
    )

    is_auth, op, err = auth.authenticate_bearer_token("Bearer secret-master-token-12345")
    assert is_auth is True
    assert err is None
    assert op is not None
    assert op.operator_id == "root_admin"
    assert "admin" in op.roles


def test_authenticator_rejects_missing_or_malformed_headers() -> None:
    """Verify missing, empty, or malformed authorization headers are rejected."""
    auth = ApprovalAuthenticator(primary_api_key="valid-key")

    # None header
    ok, op, err = auth.authenticate_bearer_token(None)
    assert ok is False
    assert "Missing Authorization header" in str(err)

    # Empty header
    ok, op, err = auth.authenticate_bearer_token("   ")
    assert ok is False
    assert "Missing Authorization header" in str(err)

    # Not Bearer
    ok, op, err = auth.authenticate_bearer_token("Token valid-key")
    assert ok is False
    assert "Malformed Authorization header" in str(err)

    # Empty token
    ok, op, err = auth.authenticate_bearer_token("Bearer   ")
    assert ok is False
    assert "cannot be empty" in str(err)


def test_authenticator_detects_impersonation() -> None:
    """Verify caller asserting a different identity than token mapping is blocked."""
    auth = ApprovalAuthenticator(
        primary_api_key="alice-key",
        default_operator="alice",
    )

    # Caller presents Alice's key but asserts 'bob'
    ok, op, err = auth.authenticate_bearer_token("Bearer alice-key", asserted_actor="bob")
    assert ok is False
    assert op is None
    assert "Impersonation detected" in str(err)

    # Matching asserted identity succeeds
    ok2, op2, err2 = auth.authenticate_bearer_token("Bearer alice-key", asserted_actor="alice")
    assert ok2 is True
    assert op2 is not None
    assert err2 is None


def test_authenticator_workflow_authorization_restrictions() -> None:
    """Verify role checks and workflow ownership boundary enforcement."""
    auth = ApprovalAuthenticator()
    alice = OperatorIdentity(operator_id="alice", roles=["admin"])
    bob = OperatorIdentity(
        operator_id="bob",
        roles=["operator"],
        allowed_workflows=["wf_bob_only"],
    )

    # Alice is admin: can authorize any workflow
    ok, err = auth.authorize_operator(alice, workflow_id="wf_anyone")
    assert ok is True
    assert err is None

    # Bob is restricted to wf_bob_only: authorized for wf_bob_only
    ok_bob, err_bob = auth.authorize_operator(bob, workflow_id="wf_bob_only")
    assert ok_bob is True
    assert err_bob is None

    # Bob is unauthorized for wf_other
    bad_bob, err_bad = auth.authorize_operator(bob, workflow_id="wf_other")
    assert bad_bob is False
    assert "not authorized to manage workflow" in str(err_bad)
