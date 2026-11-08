"""Data models and schemas for the SYNQRA Human-in-the-Loop (HITL) approval subsystem.

Defines immutable records, request parameters, cryptographically bound approval records,
and API schemas for managing human authorization decisions.
"""

import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from synqra.agents.exceptions import ApprovalError
from synqra.agents.schemas import (
    DISALLOWED_DECIDED_BY_IDENTIFIERS,
    ApprovalStatus,
    PermissionLevel,
)


def utc_now() -> datetime:
    """Return current timezone-aware UTC datetime."""
    return datetime.now(UTC)


class ApprovalDecisionRecord(BaseModel):
    """Immutable record of an authenticated human operator's decision."""

    model_config = ConfigDict(frozen=True)

    decision_id: str = Field(
        default_factory=lambda: f"dec_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this recorded decision.",
    )
    approval_id: str = Field(description="The approval_id to which this decision applies.")
    approved: bool = Field(description="True if authorized, False if rejected.")
    decided_at: datetime = Field(
        default_factory=utc_now,
        description="Timestamp when the decision was recorded.",
    )
    decided_by: str = Field(
        description="Verified identifier of the human operator rendering the decision."
    )
    reason: str | None = Field(
        default=None,
        description="Optional human-readable explanation accompanying the decision.",
    )
    decision_token: str | None = Field(
        default=None,
        description="Cryptographic HMAC binding token issued upon approval.",
    )

    @model_validator(mode="after")
    def validate_human_identity(self) -> "ApprovalDecisionRecord":
        """Verify that the decision was rendered by a human, never an automated agent/LLM."""
        actor = self.decided_by.strip().lower()
        if not actor:
            raise ApprovalError("Approval decision must provide a non-empty decided_by identifier.")
        if actor in DISALLOWED_DECIDED_BY_IDENTIFIERS:
            raise ApprovalError(
                f"Disallowed decider '{self.decided_by}': "
                "The LLM or automated agent must never approve its own operations."
            )
        return self


class ApprovalRequestRecord(BaseModel):
    """Cryptographically bound, audit-tracked approval request record."""

    approval_id: str = Field(
        default_factory=lambda: f"appr_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this approval request.",
    )
    workflow_id: str = Field(description="Identifier of the workflow requesting authorization.")
    thread_id: str = Field(
        description="LangGraph conversation thread identifier for checkpoint resumption."
    )
    step_id: str = Field(description="Specific plan step requiring authorization.")
    tool_name: str = Field(description="Exact tool targeted for execution.")
    integration_name: str = Field(
        description="Provider integration name (e.g. gmail, google_calendar, google_drive)."
    )
    description: str = Field(
        description="Human-readable explanation of exactly what the operation will do."
    )
    arguments: dict[str, Any] = Field(
        description="Exact validated arguments to be passed to the tool."
    )
    arguments_hash: str = Field(
        description="Deterministic SHA-256 hash of canonical arguments for tamper detection."
    )
    permission_level: PermissionLevel = Field(
        description="Permission tier triggering human approval (WRITE or SENSITIVE)."
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        description="Timestamp when the approval request was created.",
    )
    expires_at: datetime = Field(
        description="Timestamp after which this approval request automatically expires."
    )
    status: ApprovalStatus = Field(
        default=ApprovalStatus.PENDING,
        description="Current lifecycle status of the approval request.",
    )
    request_version: int = Field(
        default=1,
        ge=1,
        description="Monotonic version number for replay attack prevention.",
    )
    binding_token: str | None = Field(
        default=None,
        description="Cryptographic authorization token generated upon successful approval.",
    )
    decision: ApprovalDecisionRecord | None = Field(
        default=None,
        description="Associated decision record once resolved by an operator.",
    )
    consumed_at: datetime | None = Field(
        default=None,
        description="Timestamp when this approval was single-use consumed during tool execution.",
    )
    consumed_by: str | None = Field(
        default=None,
        description="Execution identifier or step that consumed this approval.",
    )

    @model_validator(mode="after")
    def validate_lifecycle_timestamps(self) -> "ApprovalRequestRecord":
        """Verify that expiration is strictly in the future relative to creation."""
        if self.expires_at <= self.created_at:
            raise ApprovalError(
                f"expires_at ({self.expires_at}) must be after created_at ({self.created_at})."
            )
        return self

    def is_expired(self, at_time: datetime | None = None) -> bool:
        """Check whether the request has expired relative to UTC now or specified time."""
        check = at_time or utc_now()
        return check >= self.expires_at


class ApprovalDecisionRequest(BaseModel):
    """API payload submitted by an authenticated human operator to approve or reject."""

    decided_by: str = Field(
        description="Verified username or identifier of the human operator.",
        min_length=1,
    )
    reason: str | None = Field(
        default=None,
        description="Optional justification or context for the decision.",
    )


class ApprovalResponse(BaseModel):
    """Structured response returned by approval API endpoints."""

    success: bool = Field(description="Whether the approval operation succeeded.")
    message: str = Field(description="Human-readable status summary.")
    approval: ApprovalRequestRecord = Field(description="Current state of the approval record.")
    decision_token: str | None = Field(
        default=None,
        description="HMAC authorization token issued if approved.",
    )


class ApprovalListResponse(BaseModel):
    """Structured response returned by the pending approvals listing endpoint."""

    total: int = Field(description="Total count of pending approval requests.")
    items: list[ApprovalRequestRecord] = Field(
        description="List of active pending approval requests."
    )
