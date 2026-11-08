"""Human-in-the-Loop (HITL) approval subsystem for SYNQRA agent workflows."""

from synqra.agents.approval.api import (
    handle_approve_request,
    handle_get_approval,
    handle_list_pending_approvals,
    handle_reject_request,
)
from synqra.agents.approval.auth import (
    ApprovalAuthenticator,
    OperatorIdentity,
    get_approval_authenticator,
    set_approval_authenticator,
)
from synqra.agents.approval.models import (
    ApprovalDecisionRecord,
    ApprovalDecisionRequest,
    ApprovalListResponse,
    ApprovalRequestRecord,
    ApprovalResponse,
)
from synqra.agents.approval.security import (
    canonical_json_serialize,
    compute_arguments_hash,
    format_human_readable_operation,
    generate_approval_binding_token,
    validate_human_actor,
    verify_approval_binding,
)
from synqra.agents.approval.service import (
    ApprovalService,
    get_approval_service,
    set_approval_service,
)
from synqra.agents.approval.sql_store import SqlAlchemyApprovalStore
from synqra.agents.approval.store import (
    BaseApprovalStore,
    InMemoryApprovalStore,
)

__all__ = [
    "ApprovalDecisionRecord",
    "ApprovalDecisionRequest",
    "ApprovalListResponse",
    "ApprovalRequestRecord",
    "ApprovalResponse",
    "ApprovalService",
    "ApprovalAuthenticator",
    "BaseApprovalStore",
    "InMemoryApprovalStore",
    "OperatorIdentity",
    "SqlAlchemyApprovalStore",
    "canonical_json_serialize",
    "compute_arguments_hash",
    "format_human_readable_operation",
    "generate_approval_binding_token",
    "get_approval_authenticator",
    "get_approval_service",
    "handle_approve_request",
    "handle_get_approval",
    "handle_list_pending_approvals",
    "handle_reject_request",
    "set_approval_authenticator",
    "set_approval_service",
    "validate_human_actor",
    "verify_approval_binding",
]
