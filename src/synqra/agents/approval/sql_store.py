"""Durable PostgreSQL and SQLite approval store using SQLAlchemy.

Provides a thread-safe, concurrent-worker-safe implementation of BaseApprovalStore
with atomic database transactions, compare-and-swap status transitions, and
automatic expiration management.
"""

import logging
from datetime import datetime
from typing import Any

from sqlalchemy import delete, select, text, update
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
)

from synqra.agents.approval.models import (
    ApprovalDecisionRecord,
    ApprovalRequestRecord,
    utc_now,
)
from synqra.agents.approval.store import BaseApprovalStore
from synqra.agents.schemas import ApprovalStatus
from synqra.database.models import ApprovalRequestModel, ensure_utc
from synqra.database.session import get_async_session_maker

logger = logging.getLogger("synqra.agents.approval.sql_store")


class SqlAlchemyApprovalStore(BaseApprovalStore):
    """PostgreSQL and SQLite durable repository for approval requests.

    Guarantees:
    1. Cross-process concurrency safety using database-level row locking and CAS transactions.
    2. Automatic expiration detection and status transition on query and update.
    3. Single-use consumption guarantees through atomic conditional updates.
    4. Full state preservation across application restarts.
    """

    def __init__(
        self,
        session_maker: async_sessionmaker[AsyncSession] | None = None,
        engine: AsyncEngine | None = None,
    ) -> None:
        if session_maker is not None:
            self._session_maker = session_maker
        elif engine is not None:
            self._session_maker = get_async_session_maker(engine=engine)
        else:
            self._session_maker = get_async_session_maker()

    async def save(self, record: ApprovalRequestRecord) -> None:
        """Persist a new approval request or update an existing one."""
        async with self._session_maker() as session, session.begin():
            stmt = select(ApprovalRequestModel).where(
                ApprovalRequestModel.approval_id == record.approval_id
            )
            res = await session.execute(stmt)
            existing = res.scalar_one_or_none()

            if existing:
                existing.workflow_id = record.workflow_id
                existing.thread_id = record.thread_id
                existing.step_id = record.step_id
                existing.tool_name = record.tool_name
                existing.integration_name = record.integration_name
                existing.description = record.description
                existing.arguments = record.arguments
                existing.arguments_hash = record.arguments_hash
                existing.permission_level = record.permission_level.value
                existing.created_at = record.created_at
                existing.expires_at = record.expires_at
                existing.status = record.status.value
                existing.request_version = record.request_version
                existing.binding_token = record.binding_token
                if record.decision:
                    existing.decision_id = record.decision.decision_id
                    existing.decision_approved = record.decision.approved
                    existing.decided_at = record.decision.decided_at
                    existing.decided_by = record.decision.decided_by
                    existing.decision_reason = record.decision.reason
                    existing.decision_token = record.decision.decision_token
                existing.consumed_at = record.consumed_at
                existing.consumed_by = record.consumed_by
            else:
                entity = ApprovalRequestModel.from_record(record)
                session.add(entity)

    async def get(self, approval_id: str) -> ApprovalRequestRecord | None:
        """Retrieve an approval record by its ID with automatic expiration check."""
        async with self._session_maker() as session:
            stmt = select(ApprovalRequestModel).where(
                ApprovalRequestModel.approval_id == approval_id
            )
            res = await session.execute(stmt)
            entity = res.scalar_one_or_none()
            if not entity:
                return None

            now = utc_now()
            # If pending and expired, transition status atomically in DB
            exp_utc = ensure_utc(entity.expires_at)
            if entity.status == ApprovalStatus.PENDING.value and exp_utc and exp_utc <= now:
                up_stmt = (
                    update(ApprovalRequestModel)
                    .where(
                        ApprovalRequestModel.approval_id == approval_id,
                        ApprovalRequestModel.status == ApprovalStatus.PENDING.value,
                    )
                    .values(status=ApprovalStatus.EXPIRED.value)
                )
                await session.execute(up_stmt)
                await session.commit()
                entity.status = ApprovalStatus.EXPIRED.value

            return entity.to_record()

    async def list_pending(self, workflow_id: str | None = None) -> list[ApprovalRequestRecord]:
        """Return non-expired pending approvals, batch-expiring past requests."""
        now = utc_now()
        async with self._session_maker() as session:
            # Batch expire any pending requests whose deadline has passed
            expire_stmt = (
                update(ApprovalRequestModel)
                .where(
                    ApprovalRequestModel.status == ApprovalStatus.PENDING.value,
                    ApprovalRequestModel.expires_at <= now,
                )
                .values(status=ApprovalStatus.EXPIRED.value)
            )
            await session.execute(expire_stmt)
            await session.commit()

            stmt = select(ApprovalRequestModel).where(
                ApprovalRequestModel.status == ApprovalStatus.PENDING.value
            )
            if workflow_id:
                stmt = stmt.where(ApprovalRequestModel.workflow_id == workflow_id)
            stmt = stmt.order_by(ApprovalRequestModel.created_at.asc())

            res = await session.execute(stmt)
            entities = res.scalars().all()
            return [e.to_record() for e in entities]

    async def list_all(self) -> list[ApprovalRequestRecord]:
        """Retrieve all approval records ordered by creation timestamp."""
        async with self._session_maker() as session:
            stmt = select(ApprovalRequestModel).order_by(ApprovalRequestModel.created_at.asc())
            res = await session.execute(stmt)
            entities = res.scalars().all()
            return [e.to_record() for e in entities]

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
        """Atomically transition status using compare-and-swap database semantics.

        Returns True if exactly one row was updated, False if concurrent conflict or expired.
        """
        now = utc_now()
        values: dict[str, Any] = {"status": new_status.value}
        if decision is not None:
            values["decision_id"] = decision.decision_id
            values["decision_approved"] = decision.approved
            values["decided_at"] = decision.decided_at
            values["decided_by"] = decision.decided_by
            values["decision_reason"] = decision.reason
            values["decision_token"] = decision.decision_token
        if binding_token is not None:
            values["binding_token"] = binding_token
        if consumed_by is not None:
            values["consumed_by"] = consumed_by
        if consumed_at is not None:
            values["consumed_at"] = consumed_at

        async with self._session_maker() as session, session.begin():
            stmt = update(ApprovalRequestModel).where(
                ApprovalRequestModel.approval_id == approval_id,
                ApprovalRequestModel.status == expected_status.value,
            )
            # If checking pending status, enforce not expired
            if expected_status == ApprovalStatus.PENDING:
                stmt = stmt.where(ApprovalRequestModel.expires_at > now)

            stmt = stmt.values(**values)
            result = await session.execute(stmt)
            return result.rowcount == 1

    async def clear(self) -> None:
        """Clear all records from the approval_requests table."""
        async with self._session_maker() as session, session.begin():
            await session.execute(delete(ApprovalRequestModel))

    async def check_health(self) -> bool:
        """Verify database connectivity by executing a lightweight connectivity query."""
        try:
            async with self._session_maker() as session:
                res = await session.execute(text("SELECT 1"))
                return res.scalar() == 1
        except Exception as exc:
            logger.error("Durable database health check failed: %s", exc)
            return False
