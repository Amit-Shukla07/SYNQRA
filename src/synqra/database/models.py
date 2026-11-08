"""Database models for SYNQRA integrations, credentials, and audit tracking."""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from synqra.agents.approval.models import (
    ApprovalDecisionRecord,
    ApprovalRequestRecord,
)
from synqra.agents.schemas import ApprovalStatus, PermissionLevel
from synqra.database.base import Base


def utc_now() -> datetime:
    """Return the current timezone-aware UTC datetime."""
    return datetime.now(UTC)


def ensure_utc(dt: datetime | None) -> datetime | None:
    """Ensure datetime has UTC timezone (normalizing SQLite naive datetimes)."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


class IntegrationCredential(Base):
    """Stores credential references and configuration status for service integrations."""

    __tablename__ = "integration_credentials"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    integration_name: Mapped[str] = mapped_column(
        String(64), unique=True, index=True, nullable=False
    )
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    auth_type: Mapped[str] = mapped_column(String(32), default="api_key", nullable=False)
    account_identifier: Mapped[str | None] = mapped_column(String(255), nullable=True)
    credentials_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    extra_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    def __repr__(self) -> str:
        return f"<IntegrationCredential(name={self.integration_name!r}, enabled={self.is_enabled})>"


class AuditLog(Base):
    """Records MCP tool calls, agent actions, and execution performance metrics."""

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str | None] = mapped_column(String(128), index=True, nullable=True)
    tool_name: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="success", nullable=False)
    execution_time_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, index=True, nullable=False
    )

    def __repr__(self) -> str:
        return f"<AuditLog(tool={self.tool_name!r}, status={self.status!r})>"


class ApprovalRequestModel(Base):
    """Persistent storage for Human-in-the-Loop approval requests and decisions."""

    __tablename__ = "approval_requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    approval_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    workflow_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    thread_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    step_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    tool_name: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    integration_name: Mapped[str] = mapped_column(String(64), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    arguments: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    arguments_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    permission_level: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), index=True, nullable=False
    )
    status: Mapped[str] = mapped_column(
        String(32), default=ApprovalStatus.PENDING.value, index=True, nullable=False
    )
    request_version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    binding_token: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Decision sub-fields
    decision_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    decision_approved: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    decided_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    decision_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    decision_token: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Consumption sub-fields
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    consumed_by: Mapped[str | None] = mapped_column(String(128), nullable=True)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    def __repr__(self) -> str:
        return f"<ApprovalRequestModel(id={self.approval_id!r}, status={self.status!r})>"

    def to_record(self) -> ApprovalRequestRecord:
        """Convert SQLAlchemy entity to strongly typed Pydantic ApprovalRequestRecord."""
        decision = None
        if self.decision_id and self.decided_by and self.decision_approved is not None:
            decision = ApprovalDecisionRecord(
                decision_id=self.decision_id,
                approval_id=self.approval_id,
                approved=self.decision_approved,
                decided_at=self.decided_at or utc_now(),
                decided_by=self.decided_by,
                reason=self.decision_reason,
                decision_token=self.decision_token,
            )

        return ApprovalRequestRecord(
            approval_id=self.approval_id,
            workflow_id=self.workflow_id,
            thread_id=self.thread_id,
            step_id=self.step_id,
            tool_name=self.tool_name,
            integration_name=self.integration_name,
            description=self.description,
            arguments=self.arguments,
            arguments_hash=self.arguments_hash,
            permission_level=PermissionLevel(self.permission_level),
            created_at=ensure_utc(self.created_at) or utc_now(),
            expires_at=ensure_utc(self.expires_at) or utc_now(),
            status=ApprovalStatus(self.status),
            request_version=self.request_version,
            binding_token=self.binding_token,
            decision=decision,
            consumed_at=ensure_utc(self.consumed_at),
            consumed_by=self.consumed_by,
        )

    @classmethod
    def from_record(cls, record: ApprovalRequestRecord) -> "ApprovalRequestModel":
        """Instantiate SQLAlchemy model from a Pydantic ApprovalRequestRecord."""
        dec = record.decision
        return cls(
            approval_id=record.approval_id,
            workflow_id=record.workflow_id,
            thread_id=record.thread_id,
            step_id=record.step_id,
            tool_name=record.tool_name,
            integration_name=record.integration_name,
            description=record.description,
            arguments=record.arguments,
            arguments_hash=record.arguments_hash,
            permission_level=record.permission_level.value,
            created_at=record.created_at,
            expires_at=record.expires_at,
            status=record.status.value,
            request_version=record.request_version,
            binding_token=record.binding_token,
            decision_id=dec.decision_id if dec else None,
            decision_approved=dec.approved if dec else None,
            decided_at=dec.decided_at if dec else None,
            decided_by=dec.decided_by if dec else None,
            decision_reason=dec.reason if dec else None,
            decision_token=dec.decision_token if dec else None,
            consumed_at=record.consumed_at,
            consumed_by=record.consumed_by,
        )
