"""Operator identity authentication and authorization for SYNQRA approvals.

Enforces server-configured bearer token validation using constant-time comparison,
supports internal SHA-256 hashed token storage, derives operator identities from
trusted server mappings, and checks workflow-level authorization.
"""

import hashlib
import hmac
import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from synqra.core.config import Settings, get_settings

logger = logging.getLogger("synqra.agents.approval.auth")

BEARER_PREFIX = "Bearer "


def normalize_token_to_hash(token: str) -> str:
    """Derive deterministic lowercase 64-char hex SHA-256 hash from a token or sha256: hash."""
    clean = token.strip()
    if clean.lower().startswith("sha256:"):
        return clean[7:].lower().strip()
    return hashlib.sha256(clean.encode("utf-8")).hexdigest()


class OperatorIdentity(BaseModel):
    """Authenticated human operator identity and role permissions."""

    operator_id: str = Field(description="Unique username or identifier of the operator.")
    roles: list[str] = Field(
        default_factory=lambda: ["operator"],
        description="Assigned RBAC roles (e.g. ['operator'], ['admin']).",
    )
    allowed_workflows: list[str] | None = Field(
        default=None,
        description="List of workflow IDs this operator is permitted to manage. None permits all.",
    )


class ApprovalAuthenticator:
    """Authenticates and authorizes operators against server-side credential configuration."""

    def __init__(
        self,
        primary_api_key: str | None = None,
        default_operator: str = "primary_operator",
        primary_roles: list[str] | None = None,
        primary_allowed_workflows: list[str] | None = None,
        token_mapping: dict[str, OperatorIdentity] | None = None,
        environment: str = "development",
    ) -> None:
        self.environment = environment
        self._token_hashes: dict[str, OperatorIdentity] = {}

        roles = primary_roles if primary_roles is not None else ["operator"]
        if primary_api_key and primary_api_key.strip():
            thash = normalize_token_to_hash(primary_api_key)
            self._token_hashes[thash] = OperatorIdentity(
                operator_id=default_operator.strip(),
                roles=roles,
                allowed_workflows=primary_allowed_workflows,
            )

        if token_mapping:
            for tok, op in token_mapping.items():
                if tok and tok.strip():
                    thash = normalize_token_to_hash(tok)
                    self._token_hashes[thash] = op

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "ApprovalAuthenticator":
        """Factory initializing authenticator directly from application Settings."""
        cfg = settings or get_settings()

        primary_key = cfg.approval_api_key.get_secret_value() if cfg.approval_api_key else None
        default_op = cfg.approval_default_operator
        primary_roles = (
            [r.strip() for r in cfg.approval_default_roles.split(",") if r.strip()]
            if cfg.approval_default_roles
            else ["operator"]
        )
        primary_wfs = (
            [w.strip() for w in cfg.approval_default_allowed_workflows.split(",") if w.strip()]
            if cfg.approval_default_allowed_workflows
            else None
        )

        token_map: dict[str, OperatorIdentity] = {}
        if cfg.approval_tokens:
            raw_tokens = cfg.approval_tokens.get_secret_value().strip()
            # Support JSON formatting or comma-separated token:operator:role list
            if raw_tokens.startswith("{"):
                try:
                    parsed: dict[str, Any] = json.loads(raw_tokens)
                    for tok_str, meta in parsed.items():
                        if isinstance(meta, dict):
                            token_map[tok_str] = OperatorIdentity(
                                operator_id=meta.get("operator_id", default_op),
                                roles=meta.get("roles", ["operator"]),
                                allowed_workflows=meta.get("allowed_workflows"),
                            )
                        elif isinstance(meta, str):
                            token_map[tok_str] = OperatorIdentity(
                                operator_id=meta,
                                roles=["operator"],
                            )
                except Exception as exc:
                    logger.error("Failed to parse SYNQRA_APPROVAL_TOKENS JSON: %s", exc)
            else:
                for entry in raw_tokens.split(","):
                    entry_clean = entry.strip()
                    if not entry_clean:
                        continue
                    if entry_clean.lower().startswith("sha256:"):
                        # Format: sha256:<hash>:<operator_id>:<roles>:<workflows>
                        parts = entry_clean.split(":")
                        if len(parts) >= 3:
                            tok_str = f"sha256:{parts[1].strip()}"
                            op_id = parts[2].strip()
                            roles = (
                                [r.strip() for r in parts[3].split("|")]
                                if len(parts) >= 4
                                else ["operator"]
                            )
                            wfs = (
                                [w.strip() for w in parts[4].split("|")]
                                if len(parts) >= 5
                                else None
                            )
                            token_map[tok_str] = OperatorIdentity(
                                operator_id=op_id,
                                roles=roles,
                                allowed_workflows=wfs,
                            )
                    else:
                        parts = entry_clean.split(":")
                        if len(parts) >= 2:
                            tok_str = parts[0].strip()
                            op_id = parts[1].strip()
                            roles = (
                                [r.strip() for r in parts[2].split("|")]
                                if len(parts) >= 3
                                else ["operator"]
                            )
                            wfs = (
                                [w.strip() for w in parts[3].split("|")]
                                if len(parts) >= 4
                                else None
                            )
                            token_map[tok_str] = OperatorIdentity(
                                operator_id=op_id,
                                roles=roles,
                                allowed_workflows=wfs,
                            )

        return cls(
            primary_api_key=primary_key,
            default_operator=default_op,
            primary_roles=primary_roles,
            primary_allowed_workflows=primary_wfs,
            token_mapping=token_map,
            environment=cfg.environment,
        )

    def register_token(
        self,
        token: str,
        operator: OperatorIdentity,
    ) -> None:
        """Register or update an in-memory operator credential mapping (hashed internally)."""
        thash = normalize_token_to_hash(token)
        self._token_hashes[thash] = operator

    def authenticate_bearer_token(
        self,
        auth_header: str | None,
        asserted_actor: str | None = None,
    ) -> tuple[bool, OperatorIdentity | None, str | None]:
        """Authenticate request credentials using constant-time token comparison.

        Enforces:
        1. Authentication must be configured; fails closed if no server secrets are set.
        2. Authorization header must follow standard `Bearer <token>` format.
        3. High entropy check: in production, requires at least 32-character tokens.
        4. Validates token against registered secrets using hmac.compare_digest on SHA-256 hashes.
        5. Detects impersonation: rejects requests where an asserted identity contradicts
           the identity bound to the authenticated token.

        Returns:
            (is_authenticated, operator_identity, failure_reason)
        """
        # 1. Fail closed if authentication credentials are not configured on server
        if not self._token_hashes:
            return (
                False,
                None,
                "Approval authentication key is not configured on the server. "
                "Set SYNQRA_APPROVAL_API_KEY in your server configuration.",
            )

        # 2. Inspect Authorization header
        if not auth_header or not auth_header.strip():
            return False, None, "Missing Authorization header. Expected Bearer token."

        parts = auth_header.strip().split(maxsplit=1)
        if len(parts) == 0:
            return False, None, "Missing Authorization header. Expected Bearer token."
        if parts[0] != "Bearer":
            return (
                False,
                None,
                "Malformed Authorization header. Format must be 'Bearer <token>'.",
            )
        if len(parts) == 1 or not parts[1].strip():
            return False, None, "Bearer token cannot be empty."

        supplied_token = parts[1].strip()

        # In production environment, enforce minimum high entropy
        if self.environment == "production" and len(supplied_token) < 32:
            return (
                False,
                None,
                (
                    "Bearer token does not meet production entropy requirements "
                    "(minimum 32 characters)."
                ),
            )

        supplied_hash = hashlib.sha256(supplied_token.encode("utf-8")).hexdigest()

        # 3. Constant-time search across configured hashes to prevent timing attacks
        matched_operator: OperatorIdentity | None = None
        for stored_hash, operator in self._token_hashes.items():
            if hmac.compare_digest(stored_hash, supplied_hash):
                matched_operator = operator
                break

        if not matched_operator:
            return False, None, "Invalid or unrecognized bearer token."

        # 4. Anti-impersonation check: if caller attempted to self-assert an actor name,
        # it must strictly match the authenticated operator identity derived from the token.
        if asserted_actor and asserted_actor.strip():
            clean_asserted = asserted_actor.strip()
            if clean_asserted != matched_operator.operator_id:
                return (
                    False,
                    None,
                    (
                        f"Impersonation detected: authenticated token belongs to "
                        f"'{matched_operator.operator_id}', but request asserted identity "
                        f"'{clean_asserted}'."
                    ),
                )

        return True, matched_operator, None

    def authorize_operator(
        self,
        operator: OperatorIdentity,
        workflow_id: str | None = None,
        required_role: str = "operator",
    ) -> tuple[bool, str | None]:
        """Verify that an authenticated operator has permission to act on the target workflow.

        Enforces role requirements and workflow ownership boundaries.
        Explicit allowed_workflows restrictions apply even to users with the admin role.
        """
        is_admin = "admin" in operator.roles
        has_role = required_role in operator.roles or is_admin

        if not has_role:
            return (
                False,
                f"Operator '{operator.operator_id}' lacks required role '{required_role}'.",
            )

        # Explicit workflow restriction takes precedence
        if (
            operator.allowed_workflows is not None
            and workflow_id
            and workflow_id not in operator.allowed_workflows
        ):
            return (
                False,
                f"Operator '{operator.operator_id}' is not authorized to manage "
                f"workflow '{workflow_id}'.",
            )

        return True, None


_default_authenticator: ApprovalAuthenticator | None = None


def get_approval_authenticator() -> ApprovalAuthenticator:
    """Return the singleton instance of the ApprovalAuthenticator."""
    global _default_authenticator
    if _default_authenticator is None:
        _default_authenticator = ApprovalAuthenticator.from_settings()
    return _default_authenticator


def set_approval_authenticator(authenticator: ApprovalAuthenticator | None) -> None:
    """Override the global authenticator singleton (used in test fixtures)."""
    global _default_authenticator
    _default_authenticator = authenticator
