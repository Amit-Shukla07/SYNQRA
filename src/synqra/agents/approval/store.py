"""Storage abstractions and thread-safe development repository for approvals.

Provides abstract storage contract and a concurrent, atomic in-memory implementation
with compare-and-swap status transitions and automatic expiration handling.
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime

from synqra.agents.approval.models import (
    ApprovalDecisionRecord,
    ApprovalRequestRecord,
    utc_now,
)
from synqra.agents.schemas import ApprovalStatus


class BaseApprovalStore(ABC):
    """Abstract interface defining required storage operations for approval records."""

    @abstractmethod
    async def save(self, record: ApprovalRequestRecord) -> None:
        """Persist a newly created approval request record."""

    @abstractmethod
    async def get(self, approval_id: str) -> ApprovalRequestRecord | None:
        """Retrieve an approval record by its ID, checking for expiration."""

    @abstractmethod
    async def list_pending(self, workflow_id: str | None = None) -> list[ApprovalRequestRecord]:
        """List active pending approval requests, optionally filtered by workflow ID."""

    @abstractmethod
    async def list_all(self) -> list[ApprovalRequestRecord]:
        """Retrieve all approval records regardless of status."""

    @abstractmethod
    async def update_status(
        self,
        approval_id: str,
        expected_status: ApprovalStatus,
        new_status: ApprovalStatus,
        decision: ApprovalDecisionRecord | None = None,
        binding_token: str | None = None,
        consumed_by: str | None = None,
        consumed_at: datetime | None = None,
    ) -> bool:
        """Atomically transition status using compare-and-swap semantics."""

    @abstractmethod
    async def clear(self) -> None:
        """Clear all records (primarily for test tear-down)."""


class InMemoryApprovalStore(BaseApprovalStore):
    """Thread-safe, atomic in-memory store for development and testing.

    IMPORTANT: Records stored in InMemoryApprovalStore do NOT survive server restarts.
    For production environments, this interface should be implemented using a durable
    relational database (e.g. PostgreSQL or SQLite via SQLAlchemy).
    """

    def __init__(self) -> None:
        self._records: dict[str, ApprovalRequestRecord] = {}
        self._lock = asyncio.Lock()

    async def save(self, record: ApprovalRequestRecord) -> None:
        """Persist or overwrite an approval record under mutex protection."""
        async with self._lock:
            self._records[record.approval_id] = record.model_copy(deep=True)

    async def get(self, approval_id: str) -> ApprovalRequestRecord | None:
        """Retrieve record by ID with automatic expiration check."""
        async with self._lock:
            record = self._records.get(approval_id)
            if not record:
                return None

            now = utc_now()
            # Automatic expiration transition
            if record.status == ApprovalStatus.PENDING and record.is_expired(now):
                updated = record.model_copy(update={"status": ApprovalStatus.EXPIRED}, deep=True)
                self._records[approval_id] = updated
                return updated.model_copy(deep=True)

            return record.model_copy(deep=True)

    async def list_pending(self, workflow_id: str | None = None) -> list[ApprovalRequestRecord]:
        """Return non-expired pending approvals."""
        async with self._lock:
            now = utc_now()
            pending: list[ApprovalRequestRecord] = []
            for app_id, rec in list(self._records.items()):
                if workflow_id and rec.workflow_id != workflow_id:
                    continue

                if rec.status == ApprovalStatus.PENDING:
                    if rec.is_expired(now):
                        # Expire in-place
                        updated = rec.model_copy(
                            update={"status": ApprovalStatus.EXPIRED}, deep=True
                        )
                        self._records[app_id] = updated
                    else:
                        pending.append(rec.model_copy(deep=True))

            return sorted(pending, key=lambda r: r.created_at)

    async def list_all(self) -> list[ApprovalRequestRecord]:
        """Return a copy of all tracked approval records."""
        async with self._lock:
            return [r.model_copy(deep=True) for r in self._records.values()]

    async def update_status(
        self,
        approval_id: str,
        expected_status: ApprovalStatus,
        new_status: ApprovalStatus,
        decision: ApprovalDecisionRecord | None = None,
        binding_token: str | None = None,
        consumed_by: str | None = None,
        consumed_at: datetime | None = None,
    ) -> bool:
        """Atomically verify current status equals expected_status and apply transition.

        Guarantees that concurrent approval or consumption attempts do not race.
        """
        async with self._lock:
            rec = self._records.get(approval_id)
            if not rec:
                return False

            now = utc_now()
            # If checking pending status, ensure it hasn't expired
            if expected_status == ApprovalStatus.PENDING and rec.is_expired(now):
                self._records[approval_id] = rec.model_copy(
                    update={"status": ApprovalStatus.EXPIRED}, deep=True
                )
                return False

            if rec.status != expected_status:
                return False

            updates: dict[str, object] = {"status": new_status}
            if decision is not None:
                updates["decision"] = decision
            if binding_token is not None:
                updates["binding_token"] = binding_token
            if consumed_by is not None:
                updates["consumed_by"] = consumed_by
            if consumed_at is not None:
                updates["consumed_at"] = consumed_at

            self._records[approval_id] = rec.model_copy(update=updates, deep=True)
            return True

    async def clear(self) -> None:
        """Clear all in-memory records."""
        async with self._lock:
            self._records.clear()


# Deferred import to preserve compatibility and prevent circular imports
def get_sqlalchemy_approval_store() -> type:
    """Return SqlAlchemyApprovalStore class."""
    from synqra.agents.approval.sql_store import SqlAlchemyApprovalStore

    return SqlAlchemyApprovalStore
