"""FastAPI and Starlette HTTP REST interface for Human-in-the-Loop approvals.

Provides authenticated endpoints for listing pending requests, inspecting exact operations,
recording human approval/rejection decisions, and enforcing fail-closed security boundaries.
"""

import logging
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from synqra.agents.approval.auth import (
    ApprovalAuthenticator,
    get_approval_authenticator,
)
from synqra.agents.approval.models import (
    ApprovalListResponse,
    ApprovalResponse,
)
from synqra.agents.approval.service import ApprovalService, get_approval_service
from synqra.agents.exceptions import ApprovalError

logger = logging.getLogger("synqra.agents.approval.api")

ACTOR_HEADER: str = "X-Synqra-Actor"


def _extract_asserted_actor(request: Request, raw_body: dict[str, Any] | None = None) -> str | None:
    """Extract any optional caller-asserted actor string for anti-impersonation checks."""
    actor = request.headers.get(ACTOR_HEADER)
    if actor and actor.strip():
        return actor.strip()
    if raw_body and isinstance(raw_body, dict):
        body_decider = raw_body.get("decided_by")
        if body_decider and str(body_decider).strip():
            return str(body_decider).strip()
    return None


async def handle_list_pending_approvals(
    request: Request,
    service: ApprovalService | None = None,
    authenticator: ApprovalAuthenticator | None = None,
) -> JSONResponse:
    """GET /api/approvals

    Lists all active pending approval requests accessible to the authenticated operator.
    """
    auth = authenticator or get_approval_authenticator()
    auth_header = request.headers.get("authorization")
    asserted = _extract_asserted_actor(request)

    is_auth, operator, auth_err = auth.authenticate_bearer_token(
        auth_header=auth_header,
        asserted_actor=asserted,
    )
    if not is_auth or not operator:
        status_code = 403 if auth_err and "impersonation" in auth_err.lower() else 401
        return JSONResponse(
            {"error": "Authentication failure", "detail": auth_err},
            status_code=status_code,
        )

    appr_service = service or get_approval_service()
    workflow_id = request.query_params.get("workflow_id")

    if workflow_id:
        is_authz, _ = auth.authorize_operator(operator, workflow_id=workflow_id)
        if not is_authz:
            return JSONResponse(
                {
                    "error": "Authorization failure",
                    "detail": "Access denied: operator is not authorized for requested workflow.",
                },
                status_code=403,
            )

    items = await appr_service.list_pending_approvals(workflow_id=workflow_id)

    # Filter items according to operator workflow boundaries if restricted
    if operator.allowed_workflows is not None:
        items = [it for it in items if it.workflow_id in operator.allowed_workflows]

    response_payload = ApprovalListResponse(total=len(items), items=items)
    return JSONResponse(response_payload.model_dump(mode="json"), status_code=200)


async def handle_get_approval(
    request: Request,
    service: ApprovalService | None = None,
    authenticator: ApprovalAuthenticator | None = None,
) -> JSONResponse:
    """GET /api/approvals/{approval_id}

    Retrieves full details and exact arguments for an approval request.
    """
    auth = authenticator or get_approval_authenticator()
    auth_header = request.headers.get("authorization")
    asserted = _extract_asserted_actor(request)

    is_auth, operator, auth_err = auth.authenticate_bearer_token(
        auth_header=auth_header,
        asserted_actor=asserted,
    )
    if not is_auth or not operator:
        status_code = 403 if auth_err and "impersonation" in auth_err.lower() else 401
        return JSONResponse(
            {"error": "Authentication failure", "detail": auth_err},
            status_code=status_code,
        )

    appr_service = service or get_approval_service()
    approval_id = request.path_params.get("approval_id", "")

    record = await appr_service.get_approval_request(approval_id)
    if not record:
        return JSONResponse(
            {"error": f"Approval request '{approval_id}' was not found."},
            status_code=404,
        )

    # Check workflow-level authorization
    is_authz, _ = auth.authorize_operator(operator, workflow_id=record.workflow_id)
    if not is_authz:
        return JSONResponse(
            {
                "error": "Authorization failure",
                "detail": "Access denied: insufficient permissions to access approval request.",
            },
            status_code=403,
        )

    return JSONResponse(record.model_dump(mode="json"), status_code=200)


async def handle_approve_request(
    request: Request,
    service: ApprovalService | None = None,
    authenticator: ApprovalAuthenticator | None = None,
) -> JSONResponse:
    """POST /api/approvals/{approval_id}/approve

    Authorizes an operation. Derives operator identity strictly from bearer credentials.
    """
    try:
        raw_body: dict[str, Any] = await request.json()
    except Exception:
        raw_body = {}

    auth = authenticator or get_approval_authenticator()
    auth_header = request.headers.get("authorization")
    asserted = _extract_asserted_actor(request, raw_body=raw_body)

    is_auth, operator, auth_err = auth.authenticate_bearer_token(
        auth_header=auth_header,
        asserted_actor=asserted,
    )
    if not is_auth or not operator:
        status_code = 403 if auth_err and "impersonation" in auth_err.lower() else 401
        return JSONResponse(
            {"error": "Authentication failure", "detail": auth_err},
            status_code=status_code,
        )

    appr_service = service or get_approval_service()
    approval_id = request.path_params.get("approval_id", "")

    record = await appr_service.get_approval_request(approval_id)
    if not record:
        return JSONResponse(
            {"error": f"Approval request '{approval_id}' was not found."},
            status_code=404,
        )

    # Check workflow-level authorization
    is_authz, _ = auth.authorize_operator(operator, workflow_id=record.workflow_id)
    if not is_authz:
        return JSONResponse(
            {
                "error": "Authorization failure",
                "detail": "Access denied: insufficient permissions to access approval request.",
            },
            status_code=403,
        )

    reason = raw_body.get("reason") if isinstance(raw_body, dict) else None

    # Derive human decider strictly from verified operator identity
    try:
        updated, token = await appr_service.approve_request(
            approval_id=approval_id,
            decided_by=operator.operator_id,
            reason=reason,
        )
    except ApprovalError as exc:
        status_code = 404 if "not found" in str(exc).lower() else 409
        return JSONResponse({"error": str(exc)}, status_code=status_code)

    resp = ApprovalResponse(
        success=True,
        message=f"Operation '{updated.tool_name}' approved by {operator.operator_id}.",
        approval=updated,
        decision_token=token,
    )
    return JSONResponse(resp.model_dump(mode="json"), status_code=200)


async def handle_reject_request(
    request: Request,
    service: ApprovalService | None = None,
    authenticator: ApprovalAuthenticator | None = None,
) -> JSONResponse:
    """POST /api/approvals/{approval_id}/reject

    Rejects an operation, permanently blocking tool execution.
    """
    try:
        raw_body: dict[str, Any] = await request.json()
    except Exception:
        raw_body = {}

    auth = authenticator or get_approval_authenticator()
    auth_header = request.headers.get("authorization")
    asserted = _extract_asserted_actor(request, raw_body=raw_body)

    is_auth, operator, auth_err = auth.authenticate_bearer_token(
        auth_header=auth_header,
        asserted_actor=asserted,
    )
    if not is_auth or not operator:
        status_code = 403 if auth_err and "impersonation" in auth_err.lower() else 401
        return JSONResponse(
            {"error": "Authentication failure", "detail": auth_err},
            status_code=status_code,
        )

    appr_service = service or get_approval_service()
    approval_id = request.path_params.get("approval_id", "")

    record = await appr_service.get_approval_request(approval_id)
    if not record:
        return JSONResponse(
            {"error": f"Approval request '{approval_id}' was not found."},
            status_code=404,
        )

    # Check workflow-level authorization
    is_authz, _ = auth.authorize_operator(operator, workflow_id=record.workflow_id)
    if not is_authz:
        return JSONResponse(
            {
                "error": "Authorization failure",
                "detail": "Access denied: insufficient permissions to access approval request.",
            },
            status_code=403,
        )

    reason = raw_body.get("reason") if isinstance(raw_body, dict) else None

    try:
        updated = await appr_service.reject_request(
            approval_id=approval_id,
            decided_by=operator.operator_id,
            reason=reason,
        )
    except ApprovalError as exc:
        status_code = 404 if "not found" in str(exc).lower() else 409
        return JSONResponse({"error": str(exc)}, status_code=status_code)

    resp = ApprovalResponse(
        success=True,
        message=f"Operation '{updated.tool_name}' rejected by {operator.operator_id}.",
        approval=updated,
    )
    return JSONResponse(resp.model_dump(mode="json"), status_code=200)
