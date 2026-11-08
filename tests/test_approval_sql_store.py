"""Integration tests for SqlAlchemyApprovalStore.

Verifies durable PostgreSQL/SQLite storage of approval requests, compare-and-swap (CAS)
status transitions, automatic expiration management, and concurrency safety.
"""

import asyncio
from datetime import timedelta

import pytest
from sqlalchemy.ext.asyncio import (
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
from synqra.agents.exceptions import ApprovalError
from synqra.agents.schemas import ApprovalStatus, PermissionLevel
from synqra.database.base import Base


@pytest.fixture
async def sql_store() -> SqlAlchemyApprovalStore:
    """Fixture providing an isolated SQLite in-memory database and SqlAlchemyApprovalStore."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        future=True,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_maker = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )
    store = SqlAlchemyApprovalStore(session_maker=session_maker)
    yield store
    await engine.dispose()


@pytest.mark.asyncio
async def test_save_and_get_approval_request(sql_store: SqlAlchemyApprovalStore) -> None:
    """Verify persisting and retrieving an approval request from durable storage."""
    rec = ApprovalRequestRecord(
        workflow_id="wf_sql_1",
        thread_id="th_sql_1",
        step_id="step_sql_1",
        tool_name="upload_drive_file",
        integration_name="google_drive",
        description="Upload audit report",
        arguments={"filename": "audit.pdf"},
        arguments_hash="hash_12345",
        permission_level=PermissionLevel.WRITE,
        expires_at=utc_now() + timedelta(minutes=15),
    )

    await sql_store.save(rec)

    retrieved = await sql_store.get(rec.approval_id)
    assert retrieved is not None
    assert retrieved.approval_id == rec.approval_id
    assert retrieved.workflow_id == "wf_sql_1"
    assert retrieved.step_id == "step_sql_1"
    assert retrieved.tool_name == "upload_drive_file"
    assert retrieved.arguments == {"filename": "audit.pdf"}
    assert retrieved.status == ApprovalStatus.PENDING
    assert retrieved.decision is None


@pytest.mark.asyncio
async def test_get_nonexistent_returns_none(sql_store: SqlAlchemyApprovalStore) -> None:
    """Verify querying non-existent approval returns None."""
    retrieved = await sql_store.get("appr_nonexistent_999")
    assert retrieved is None


@pytest.mark.asyncio
async def test_automatic_expiration_on_get(sql_store: SqlAlchemyApprovalStore) -> None:
    """Verify that an expired pending request transitions to EXPIRED in database on get."""
    # Create request that expired 10 seconds ago
    now = utc_now()
    rec = ApprovalRequestRecord(
        workflow_id="wf_exp",
        thread_id="th_exp",
        step_id="step_exp",
        tool_name="send_gmail_message",
        integration_name="gmail",
        description="Send email",
        arguments={"to": "user@corp.com"},
        arguments_hash="hash_abc",
        permission_level=PermissionLevel.SENSITIVE,
        created_at=now - timedelta(minutes=10),
        expires_at=now - timedelta(seconds=10),
    )

    await sql_store.save(rec)

    # Retrieval triggers automatic expiration
    retrieved = await sql_store.get(rec.approval_id)
    assert retrieved is not None
    assert retrieved.status == ApprovalStatus.EXPIRED

    # Direct query confirms persisted status in database is EXPIRED
    pending = await sql_store.list_pending()
    assert len(pending) == 0


@pytest.mark.asyncio
async def test_list_pending_filtering_and_sorting(
    sql_store: SqlAlchemyApprovalStore,
) -> None:
    """Verify listing pending approvals with workflow scoping and created_at ordering."""
    now = utc_now()

    # Rec 1: wf_alpha, created first
    r1 = ApprovalRequestRecord(
        workflow_id="wf_alpha",
        thread_id="th_1",
        step_id="step_1",
        tool_name="tool_1",
        integration_name="drive",
        description="Step 1",
        arguments={},
        arguments_hash="h1",
        permission_level=PermissionLevel.WRITE,
        created_at=now,
        expires_at=now + timedelta(minutes=10),
    )
    # Rec 2: wf_beta, created second
    r2 = ApprovalRequestRecord(
        workflow_id="wf_beta",
        thread_id="th_2",
        step_id="step_2",
        tool_name="tool_2",
        integration_name="gmail",
        description="Step 2",
        arguments={},
        arguments_hash="h2",
        permission_level=PermissionLevel.SENSITIVE,
        created_at=now + timedelta(seconds=5),
        expires_at=now + timedelta(minutes=10),
    )
    # Rec 3: wf_alpha, created third
    r3 = ApprovalRequestRecord(
        workflow_id="wf_alpha",
        thread_id="th_3",
        step_id="step_3",
        tool_name="tool_3",
        integration_name="drive",
        description="Step 3",
        arguments={},
        arguments_hash="h3",
        permission_level=PermissionLevel.WRITE,
        created_at=now + timedelta(seconds=10),
        expires_at=now + timedelta(minutes=10),
    )

    await sql_store.save(r1)
    await sql_store.save(r2)
    await sql_store.save(r3)

    # 1. Unfiltered list returns all pending in chronological order
    all_pending = await sql_store.list_pending()
    assert len(all_pending) == 3
    assert [r.approval_id for r in all_pending] == [
        r1.approval_id,
        r2.approval_id,
        r3.approval_id,
    ]

    # 2. Scoped by workflow_id
    alpha_pending = await sql_store.list_pending(workflow_id="wf_alpha")
    assert len(alpha_pending) == 2
    assert [r.approval_id for r in alpha_pending] == [r1.approval_id, r3.approval_id]

    beta_pending = await sql_store.list_pending(workflow_id="wf_beta")
    assert len(beta_pending) == 1
    assert beta_pending[0].approval_id == r2.approval_id


@pytest.mark.asyncio
async def test_atomic_status_update_cas(sql_store: SqlAlchemyApprovalStore) -> None:
    """Verify compare-and-swap status transition semantics."""
    rec = ApprovalRequestRecord(
        workflow_id="wf_cas",
        thread_id="th_cas",
        step_id="step_cas",
        tool_name="upload_drive_file",
        integration_name="google_drive",
        description="CAS test",
        arguments={"q": "1"},
        arguments_hash="h_cas",
        permission_level=PermissionLevel.WRITE,
        expires_at=utc_now() + timedelta(minutes=10),
    )
    await sql_store.save(rec)

    # 1. Successful PENDING -> APPROVED update
    decision = ApprovalDecisionRecord(
        approval_id=rec.approval_id,
        approved=True,
        decided_by="admin_alice",
        reason="Looks safe",
        decision_token="token_hmac_123",
    )
    success = await sql_store.update_status(
        approval_id=rec.approval_id,
        expected_status=ApprovalStatus.PENDING,
        new_status=ApprovalStatus.APPROVED,
        decision=decision,
        binding_token="binding_token_456",
    )
    assert success is True

    updated = await sql_store.get(rec.approval_id)
    assert updated is not None
    assert updated.status == ApprovalStatus.APPROVED
    assert updated.decision is not None
    assert updated.decision.approved is True
    assert updated.decision.decided_by == "admin_alice"
    assert updated.binding_token == "binding_token_456"

    # 2. Conflicting CAS update fails: expected PENDING but current is APPROVED
    conflict = await sql_store.update_status(
        approval_id=rec.approval_id,
        expected_status=ApprovalStatus.PENDING,
        new_status=ApprovalStatus.REJECTED,
    )
    assert conflict is False

    # 3. Successful APPROVED -> CONSUMED update
    now = utc_now()
    consume_ok = await sql_store.update_status(
        approval_id=rec.approval_id,
        expected_status=ApprovalStatus.APPROVED,
        new_status=ApprovalStatus.CONSUMED,
        consumed_by="wf_cas:step_cas",
        consumed_at=now,
    )
    assert consume_ok is True

    consumed = await sql_store.get(rec.approval_id)
    assert consumed is not None
    assert consumed.status == ApprovalStatus.CONSUMED
    assert consumed.consumed_by == "wf_cas:step_cas"

    # 4. Replay consume fails
    re_consume = await sql_store.update_status(
        approval_id=rec.approval_id,
        expected_status=ApprovalStatus.APPROVED,
        new_status=ApprovalStatus.CONSUMED,
    )
    assert re_consume is False


@pytest.mark.asyncio
async def test_concurrent_worker_approval_race(
    sql_store: SqlAlchemyApprovalStore,
) -> None:
    """Verify cross-worker concurrency safety: two simultaneous approve calls cannot both win."""
    rec = ApprovalRequestRecord(
        workflow_id="wf_race_sql",
        thread_id="th_race_sql",
        step_id="step_race_sql",
        tool_name="send_gmail_message",
        integration_name="gmail",
        description="Race condition test",
        arguments={"q": "race"},
        arguments_hash="h_race",
        permission_level=PermissionLevel.SENSITIVE,
        expires_at=utc_now() + timedelta(minutes=10),
    )
    await sql_store.save(rec)

    async def _worker_approve(worker_id: str) -> bool:
        decision = ApprovalDecisionRecord(
            approval_id=rec.approval_id,
            approved=True,
            decided_by=worker_id,
        )
        return await sql_store.update_status(
            approval_id=rec.approval_id,
            expected_status=ApprovalStatus.PENDING,
            new_status=ApprovalStatus.APPROVED,
            decision=decision,
        )

    # Launch two simultaneous workers competing to approve
    w1, w2 = await asyncio.gather(
        _worker_approve("worker_1"),
        _worker_approve("worker_2"),
    )

    # Exactly one worker wins; one fails
    assert sorted([w1, w2]) == [False, True]

    # Verify database state reflects the winning worker
    final_rec = await sql_store.get(rec.approval_id)
    assert final_rec is not None
    assert final_rec.status == ApprovalStatus.APPROVED
    assert final_rec.decision is not None
    assert final_rec.decision.decided_by in ("worker_1", "worker_2")


@pytest.mark.asyncio
async def test_concurrent_worker_consumption_race(
    sql_store: SqlAlchemyApprovalStore,
) -> None:
    """Verify that two concurrent workers cannot both consume an approved record."""
    rec = ApprovalRequestRecord(
        workflow_id="wf_consume_race_sql",
        thread_id="th_consume_race_sql",
        step_id="step_consume",
        tool_name="upload_drive_file",
        integration_name="google_drive",
        description="Consume race test",
        arguments={"q": "consume"},
        arguments_hash="h_consume",
        permission_level=PermissionLevel.WRITE,
        status=ApprovalStatus.APPROVED,
        expires_at=utc_now() + timedelta(minutes=10),
    )
    await sql_store.save(rec)

    async def _worker_consume(worker_id: str) -> bool:
        return await sql_store.update_status(
            approval_id=rec.approval_id,
            expected_status=ApprovalStatus.APPROVED,
            new_status=ApprovalStatus.CONSUMED,
            consumed_by=worker_id,
            consumed_at=utc_now(),
        )

    # Two workers attempt single-use consumption at the exact same instant
    res1, res2 = await asyncio.gather(
        _worker_consume("worker_A"),
        _worker_consume("worker_B"),
    )

    assert sorted([res1, res2]) == [False, True]

    consumed_rec = await sql_store.get(rec.approval_id)
    assert consumed_rec is not None
    assert consumed_rec.status == ApprovalStatus.CONSUMED
    assert consumed_rec.consumed_by in ("worker_A", "worker_B")


@pytest.mark.asyncio
async def test_approval_service_with_sql_store(
    sql_store: SqlAlchemyApprovalStore,
) -> None:
    """Verify full end-to-end integration between ApprovalService and SqlAlchemyApprovalStore."""
    service = ApprovalService(store=sql_store)

    # 1. Create request
    args = {"recipient": "client@corp.com", "subject": "Q4 Results"}
    created = await service.create_approval_request(
        workflow_id="wf_svc_sql",
        thread_id="th_svc_sql",
        step_id="step_svc",
        tool_name="send_gmail_message",
        arguments=args,
        permission_level=PermissionLevel.SENSITIVE,
        description="Send Q4 financial update",
    )
    assert created.status == ApprovalStatus.PENDING

    # 2. Approve request
    approved, token = await service.approve_request(
        approval_id=created.approval_id,
        decided_by="executive_carol",
        reason="Authorized for board release",
    )
    assert approved.status == ApprovalStatus.APPROVED
    assert approved.decision is not None
    assert approved.decision.decided_by == "executive_carol"
    assert token is not None

    # 3. Duplicate approval fails
    with pytest.raises(ApprovalError, match="cannot be approved: status is approved"):
        await service.approve_request(
            approval_id=created.approval_id,
            decided_by="executive_dan",
        )

    # 4. Consume approval for tool execution
    ok_consume, consume_err = await service.consume_approval(
        approval_id=created.approval_id,
        workflow_id="wf_svc_sql",
        thread_id="th_svc_sql",
        step_id="step_svc",
        tool_name="send_gmail_message",
        arguments=args,
        permission_level=PermissionLevel.SENSITIVE,
    )
    assert ok_consume is True
    assert consume_err is None

    # 5. Replay consume fails
    ok_replay, replay_err = await service.consume_approval(
        approval_id=created.approval_id,
        workflow_id="wf_svc_sql",
        thread_id="th_svc_sql",
        step_id="step_svc",
        tool_name="send_gmail_message",
        arguments=args,
        permission_level=PermissionLevel.SENSITIVE,
    )
    assert ok_replay is False
    assert "already been consumed" in str(replay_err).lower()


def test_service_factory_store_selection_no_silent_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify selecting database mode uses SqlAlchemyApprovalStore and never silently falls back."""
    from synqra.core.config import get_settings

    # 1. Default memory mode
    monkeypatch.setenv("SYNQRA_APPROVAL_STORE_TYPE", "memory")
    get_settings.cache_clear()
    svc_mem = ApprovalService()
    from synqra.agents.approval.store import InMemoryApprovalStore

    assert isinstance(svc_mem.store, InMemoryApprovalStore)

    # 2. Database mode explicitly selects SqlAlchemyApprovalStore
    monkeypatch.setenv("SYNQRA_APPROVAL_STORE_TYPE", "database")
    get_settings.cache_clear()
    svc_db = ApprovalService()
    assert isinstance(svc_db.store, SqlAlchemyApprovalStore)

    # Reset cache
    get_settings.cache_clear()
