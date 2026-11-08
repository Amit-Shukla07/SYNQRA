"""Live PostgreSQL Integration Tests for SYNQRA Approval Repository (Phase 9.5.1.1).

Executes comprehensive end-to-end persistence, concurrency, and CAS tests against
an actual running PostgreSQL database (PostgreSQL 16 via psycopg).

These tests are clearly distinguished from in-memory SQLite development tests.
"""

import asyncio
import contextlib
import os
from collections.abc import AsyncGenerator
from datetime import timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from synqra.agents.approval.models import (
    ApprovalDecisionRecord,
    ApprovalRequestRecord,
    utc_now,
)
from synqra.agents.approval.service import ApprovalService
from synqra.agents.approval.sql_store import SqlAlchemyApprovalStore
from synqra.agents.schemas import ApprovalStatus, PermissionLevel
from synqra.database.base import Base

DEFAULT_POSTGRES_TEST_URL = "postgresql+psycopg://amitshukla@localhost:5432/synqra_test"


async def check_postgres_available(url: str) -> bool:
    """Probe if PostgreSQL is accessible at the specified connection string."""
    try:
        engine = create_async_engine(url)
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1;"))
        await engine.dispose()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def postgres_url() -> str:
    """Return the configured PostgreSQL test URL."""
    return os.getenv("SYNQRA_TEST_POSTGRES_URL", DEFAULT_POSTGRES_TEST_URL)


@pytest.fixture
async def live_pg_store(postgres_url: str) -> AsyncGenerator[SqlAlchemyApprovalStore, None]:
    """Provide a clean live PostgreSQL approval store for each test."""
    is_available = await check_postgres_available(postgres_url)
    if not is_available:
        pytest.skip(
            f"Live PostgreSQL server is not available at {postgres_url}. "
            "Set SYNQRA_TEST_POSTGRES_URL to run live PostgreSQL verification."
        )

    engine: AsyncEngine = create_async_engine(
        postgres_url,
        echo=False,
        future=True,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
    )

    # Ensure tables exist on test database
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_maker = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )
    store = SqlAlchemyApprovalStore(session_maker=session_maker)

    # Clean test table before running
    await store.clear()

    yield store

    # Clean test table after test and dispose engine
    with contextlib.suppress(Exception):
        await store.clear()
    await engine.dispose()


# ---------------------------------------------------------------------------
# Live PostgreSQL Verification Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_postgres_live_save_and_retrieve_in_new_session(
    live_pg_store: SqlAlchemyApprovalStore,
) -> None:
    """Verify saving an approval request in PostgreSQL and retrieving it in a fresh session."""
    now = utc_now()
    rec = ApprovalRequestRecord(
        workflow_id="wf_pg_1",
        thread_id="th_pg_1",
        step_id="step_pg_1",
        tool_name="upload_drive_file",
        integration_name="google_drive",
        description="Upload executive financial statement",
        arguments={"filename": "Q3_results.pdf", "folder_id": "fld_999"},
        arguments_hash="hash_pg_sec_123",
        permission_level=PermissionLevel.WRITE,
        created_at=now,
        expires_at=now + timedelta(minutes=15),
    )

    # 1. Save in PostgreSQL
    await live_pg_store.save(rec)

    # 2. Retrieve in an entirely new session
    retrieved = await live_pg_store.get(rec.approval_id)
    assert retrieved is not None
    assert retrieved.approval_id == rec.approval_id
    assert retrieved.workflow_id == "wf_pg_1"
    assert retrieved.tool_name == "upload_drive_file"
    assert retrieved.arguments == {"filename": "Q3_results.pdf", "folder_id": "fld_999"}
    assert retrieved.status == ApprovalStatus.PENDING

    # 3. Ensure timestamps remain timezone-aware UTC values in PostgreSQL
    assert retrieved.created_at.tzinfo is not None
    assert retrieved.expires_at.tzinfo is not None


@pytest.mark.asyncio
async def test_postgres_live_update_status(
    live_pg_store: SqlAlchemyApprovalStore,
) -> None:
    """Verify conditional status transition in PostgreSQL."""
    rec = ApprovalRequestRecord(
        workflow_id="wf_pg_update",
        thread_id="th_pg_update",
        step_id="step_update",
        tool_name="create_calendar_event",
        integration_name="google_calendar",
        description="Schedule security review",
        arguments={"summary": "Security Review"},
        arguments_hash="hash_update_456",
        permission_level=PermissionLevel.WRITE,
        expires_at=utc_now() + timedelta(minutes=10),
    )
    await live_pg_store.save(rec)

    # CAS update from PENDING to APPROVED
    decision = ApprovalDecisionRecord(
        approval_id=rec.approval_id,
        approved=True,
        decided_by="admin_sec_lead",
        reason="Verified meeting participants",
    )
    updated = await live_pg_store.update_status(
        approval_id=rec.approval_id,
        expected_status=ApprovalStatus.PENDING,
        new_status=ApprovalStatus.APPROVED,
        decision=decision,
        binding_token="hmac_binding_token_pg",
    )
    assert updated is True

    # Conflicting update fails
    conflict = await live_pg_store.update_status(
        approval_id=rec.approval_id,
        expected_status=ApprovalStatus.PENDING,
        new_status=ApprovalStatus.REJECTED,
    )
    assert conflict is False


@pytest.mark.asyncio
async def test_postgres_live_approve_and_consume_request(
    live_pg_store: SqlAlchemyApprovalStore,
) -> None:
    """Verify full approval and single-use consumption lifecycle in PostgreSQL."""
    service = ApprovalService(store=live_pg_store)
    args = {"to": "board@corp.com", "subject": "Audit Signoff"}

    # 1. Create request
    created = await service.create_approval_request(
        workflow_id="wf_pg_full",
        thread_id="th_pg_full",
        step_id="step_signoff",
        tool_name="send_gmail_message",
        arguments=args,
        permission_level=PermissionLevel.SENSITIVE,
        description="Send board signoff email",
    )
    assert created.status == ApprovalStatus.PENDING

    # 2. Approve request
    approved, token = await service.approve_request(
        approval_id=created.approval_id,
        decided_by="ciso_operator",
        reason="Cryptographically approved by CISO",
    )
    assert approved.status == ApprovalStatus.APPROVED
    assert approved.decision is not None
    assert approved.decision.decided_by == "ciso_operator"
    assert token is not None

    # 3. Consume approval
    ok_consume, err_consume = await service.consume_approval(
        approval_id=created.approval_id,
        workflow_id="wf_pg_full",
        thread_id="th_pg_full",
        step_id="step_signoff",
        tool_name="send_gmail_message",
        arguments=args,
        permission_level=PermissionLevel.SENSITIVE,
    )
    assert ok_consume is True
    assert err_consume is None

    # 4. Replay consume fails
    ok_replay, err_replay = await service.consume_approval(
        approval_id=created.approval_id,
        workflow_id="wf_pg_full",
        thread_id="th_pg_full",
        step_id="step_signoff",
        tool_name="send_gmail_message",
        arguments=args,
        permission_level=PermissionLevel.SENSITIVE,
    )
    assert ok_replay is False
    assert "already been consumed" in str(err_replay).lower()


@pytest.mark.asyncio
async def test_postgres_live_reject_request(
    live_pg_store: SqlAlchemyApprovalStore,
) -> None:
    """Verify human rejection is persisted durably in PostgreSQL."""
    service = ApprovalService(store=live_pg_store)
    rec = await service.create_approval_request(
        workflow_id="wf_pg_rej",
        thread_id="th_pg_rej",
        step_id="step_delete",
        tool_name="delete_calendar_event",
        arguments={"event_id": "ev_important_1"},
        permission_level=PermissionLevel.WRITE,
    )

    rejected = await service.reject_request(
        approval_id=rec.approval_id,
        decided_by="manager_dan",
        reason="Deletion not authorized",
    )
    assert rejected.status == ApprovalStatus.REJECTED
    assert rejected.decision is not None
    assert rejected.decision.approved is False
    assert rejected.decision.decided_by == "manager_dan"


@pytest.mark.asyncio
async def test_postgres_live_automatic_expiration(
    live_pg_store: SqlAlchemyApprovalStore,
) -> None:
    """Verify that requests past their expires_at are auto-expired in PostgreSQL."""
    now = utc_now()
    past_rec = ApprovalRequestRecord(
        workflow_id="wf_pg_exp",
        thread_id="th_pg_exp",
        step_id="step_exp",
        tool_name="upload_drive_file",
        integration_name="google_drive",
        description="Expired upload",
        arguments={"file": "old.txt"},
        arguments_hash="hash_old",
        permission_level=PermissionLevel.WRITE,
        created_at=now - timedelta(minutes=20),
        expires_at=now - timedelta(minutes=5),
    )
    await live_pg_store.save(past_rec)

    # Retrieval in PostgreSQL triggers status transition to EXPIRED
    retrieved = await live_pg_store.get(past_rec.approval_id)
    assert retrieved is not None
    assert retrieved.status == ApprovalStatus.EXPIRED

    # Pending query does not return expired items
    pending = await live_pg_store.list_pending(workflow_id="wf_pg_exp")
    assert len(pending) == 0


@pytest.mark.asyncio
async def test_postgres_live_persistence_across_connection_reopen(
    postgres_url: str,
) -> None:
    """Verify records persist in PostgreSQL after completely closing and reopening connection."""
    is_available = await check_postgres_available(postgres_url)
    if not is_available:
        pytest.skip(f"PostgreSQL not reachable at {postgres_url}")

    # 1. Initialize first engine and store
    engine1 = create_async_engine(postgres_url, future=True)
    session_maker1 = async_sessionmaker(bind=engine1, class_=AsyncSession, expire_on_commit=False)
    store1 = SqlAlchemyApprovalStore(session_maker=session_maker1)

    rec = ApprovalRequestRecord(
        workflow_id="wf_pg_persist",
        thread_id="th_pg_persist",
        step_id="step_persist",
        tool_name="upload_drive_file",
        integration_name="google_drive",
        description="Persistence across restart test",
        arguments={"file": "durability_proof.txt"},
        arguments_hash="hash_persist_999",
        permission_level=PermissionLevel.WRITE,
        expires_at=utc_now() + timedelta(minutes=30),
    )
    await store1.save(rec)

    # 2. Simulate complete application shutdown: dispose engine and tear down connection pool
    await engine1.dispose()

    # 3. Simulate application restart: establish completely fresh engine and session maker
    engine2 = create_async_engine(postgres_url, future=True)
    session_maker2 = async_sessionmaker(bind=engine2, class_=AsyncSession, expire_on_commit=False)
    store2 = SqlAlchemyApprovalStore(session_maker=session_maker2)

    try:
        # Retrieve record created before restart
        retrieved = await store2.get(rec.approval_id)
        assert retrieved is not None
        assert retrieved.approval_id == rec.approval_id
        assert retrieved.workflow_id == "wf_pg_persist"
        assert retrieved.arguments == {"file": "durability_proof.txt"}
        assert retrieved.status == ApprovalStatus.PENDING
    finally:
        await store2.clear()
        await engine2.dispose()


@pytest.mark.asyncio
async def test_postgres_live_concurrent_approvals(
    live_pg_store: SqlAlchemyApprovalStore,
) -> None:
    """Verify cross-worker concurrency: 10 concurrent approval calls in PostgreSQL."""
    rec = ApprovalRequestRecord(
        workflow_id="wf_pg_conc_appr",
        thread_id="th_pg_conc_appr",
        step_id="step_conc_appr",
        tool_name="upload_drive_file",
        integration_name="google_drive",
        description="Concurrency test for approval in PG",
        arguments={"q": "pg_race"},
        arguments_hash="hash_pg_race",
        permission_level=PermissionLevel.WRITE,
        expires_at=utc_now() + timedelta(minutes=10),
    )
    await live_pg_store.save(rec)

    async def _worker_approve(worker_id: str) -> bool:
        decision = ApprovalDecisionRecord(
            approval_id=rec.approval_id,
            approved=True,
            decided_by=f"operator_{worker_id}",
        )
        return await live_pg_store.update_status(
            approval_id=rec.approval_id,
            expected_status=ApprovalStatus.PENDING,
            new_status=ApprovalStatus.APPROVED,
            decision=decision,
        )

    # 10 concurrent workers compete to approve
    results = await asyncio.gather(*[_worker_approve(str(i)) for i in range(10)])

    # Exactly 1 worker wins; 9 fail cleanly
    assert results.count(True) == 1
    assert results.count(False) == 9

    final = await live_pg_store.get(rec.approval_id)
    assert final is not None
    assert final.status == ApprovalStatus.APPROVED


@pytest.mark.asyncio
async def test_postgres_live_concurrent_consumption_race(
    live_pg_store: SqlAlchemyApprovalStore,
) -> None:
    """Verify that 10 concurrent workers attempting to consume an approval execute exactly once."""
    rec = ApprovalRequestRecord(
        workflow_id="wf_pg_conc_consume",
        thread_id="th_pg_conc_consume",
        step_id="step_conc_consume",
        tool_name="send_gmail_message",
        integration_name="gmail",
        description="Concurrent consumption in PG",
        arguments={"recipient": "boss@corp.com"},
        arguments_hash="hash_consume_pg",
        permission_level=PermissionLevel.SENSITIVE,
        status=ApprovalStatus.APPROVED,
        expires_at=utc_now() + timedelta(minutes=10),
    )
    await live_pg_store.save(rec)

    async def _worker_consume(worker_id: str) -> bool:
        return await live_pg_store.update_status(
            approval_id=rec.approval_id,
            expected_status=ApprovalStatus.APPROVED,
            new_status=ApprovalStatus.CONSUMED,
            consumed_by=f"worker_process_{worker_id}",
            consumed_at=utc_now(),
        )

    # 10 workers simultaneously attempt consumption
    results = await asyncio.gather(*[_worker_consume(str(i)) for i in range(10)])

    # Exactly 1 succeeds; 9 fail
    assert results.count(True) == 1
    assert results.count(False) == 9

    final = await live_pg_store.get(rec.approval_id)
    assert final is not None
    assert final.status == ApprovalStatus.CONSUMED


@pytest.mark.asyncio
async def test_postgres_live_duplicate_approval_id_rejected(
    live_pg_store: SqlAlchemyApprovalStore,
) -> None:
    """Verify PostgreSQL unique constraint on approval_id enforces integrity."""
    from synqra.database.models import ApprovalRequestModel

    rec = ApprovalRequestRecord(
        approval_id="appr_duplicate_test_123",
        workflow_id="wf_dup",
        thread_id="th_dup",
        step_id="step_dup",
        tool_name="upload_drive_file",
        integration_name="google_drive",
        description="Duplicate ID test",
        arguments={"q": "dup"},
        arguments_hash="h_dup",
        permission_level=PermissionLevel.WRITE,
        expires_at=utc_now() + timedelta(minutes=10),
    )
    await live_pg_store.save(rec)

    # Attempting to insert a duplicate approval_id directly into PostgreSQL raises IntegrityError
    with pytest.raises(IntegrityError):
        async with live_pg_store._session_maker() as session, session.begin():
            dup_entity = ApprovalRequestModel.from_record(rec)
            session.add(dup_entity)


@pytest.mark.asyncio
async def test_postgres_live_health_check(
    live_pg_store: SqlAlchemyApprovalStore,
) -> None:
    """Verify check_health executes successfully against live PostgreSQL."""
    is_healthy = await live_pg_store.check_health()
    assert is_healthy is True
