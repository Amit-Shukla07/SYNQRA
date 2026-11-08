"""Tests for PostgreSQL-backed LangGraph Checkpointer (Phase 9.5.2).

Validates connection normalization, credential masking, AsyncPostgresSaver setup,
checkpoint creation, retrieval, thread isolation, and fail-closed error handling
against live PostgreSQL.
"""

from __future__ import annotations

import uuid

import pytest
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from synqra.agents.checkpointer import (
    PostgresCheckpointerManager,
    get_checkpointer,
    mask_database_url,
    normalize_postgres_url,
)
from synqra.agents.exceptions import (
    CheckpointerConfigurationError,
    CheckpointerConnectionError,
)
from synqra.agents.graph import create_agent_graph
from synqra.agents.llm import MockChatModel
from synqra.agents.schemas import ExecutionPlan, PlanStep, WorkflowStatus

POSTGRES_TEST_URL = "postgresql://amitshukla@localhost:5432/synqra_test"


async def check_postgres_alive() -> bool:
    try:
        pool = AsyncConnectionPool(POSTGRES_TEST_URL, open=False, min_size=1, max_size=2)
        await pool.open(wait=True)
        await pool.close()
        return True
    except Exception:
        return False


def test_normalize_postgres_url() -> None:
    """Verify URL normalization removes driver prefixes correctly."""
    assert (
        normalize_postgres_url("postgresql+psycopg://user:pass@localhost:5432/db")
        == "postgresql://user:pass@localhost:5432/db"
    )
    assert (
        normalize_postgres_url("postgresql+asyncpg://user:pass@localhost:5432/db")
        == "postgresql://user:pass@localhost:5432/db"
    )
    assert (
        normalize_postgres_url("postgresql://user:pass@localhost:5432/db")
        == "postgresql://user:pass@localhost:5432/db"
    )

    with pytest.raises(CheckpointerConfigurationError):
        normalize_postgres_url("sqlite+aiosqlite:///test.db")

    with pytest.raises(CheckpointerConfigurationError):
        normalize_postgres_url("")


def test_mask_database_url() -> None:
    """Verify passwords in connection URLs are masked."""
    masked = mask_database_url("postgresql://synqra_admin:supersecret@db.internal:5432/synqra_prod")
    assert "supersecret" not in masked
    assert ":***@" in masked
    assert "db.internal:5432/synqra_prod" in masked


@pytest.mark.asyncio
async def test_postgres_checkpointer_manager_health() -> None:
    """Test PostgresCheckpointerManager health reporting."""
    alive = await check_postgres_alive()
    if not alive:
        pytest.skip("PostgreSQL test database not available.")

    manager = PostgresCheckpointerManager()
    assert not manager.is_initialized

    # Health before init
    h1 = await manager.check_health()
    assert h1["healthy"] is False

    # Initialize
    saver = await manager.initialize(database_url=POSTGRES_TEST_URL, min_size=1, max_size=3)
    assert manager.is_initialized
    assert saver is not None

    h2 = await manager.check_health()
    assert h2["healthy"] is True
    assert h2["status"] == "connected"

    await manager.close()
    assert not manager.is_initialized


@pytest.mark.asyncio
async def test_postgres_checkpoint_creation_and_retrieval() -> None:
    """Test checkpoint creation and retrieval on live PostgreSQL."""
    alive = await check_postgres_alive()
    if not alive:
        pytest.skip("PostgreSQL test database not available.")

    manager = PostgresCheckpointerManager()
    saver = await manager.initialize(database_url=POSTGRES_TEST_URL, min_size=1, max_size=3)

    tid = f"th_test_create_{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": tid, "checkpoint_ns": ""}}

    # Put a state checkpoint
    test_checkpoint = {
        "v": 1,
        "ts": "2026-11-08T12:00:00.000000+00:00",
        "id": str(uuid.uuid4()),
        "channel_values": {
            "workflow_id": "wf_test_1",
            "thread_id": tid,
            "workflow_status": "executing",
        },
        "channel_versions": {"workflow_id": 1, "thread_id": 1, "workflow_status": 1},
        "versions_seen": {},
    }
    await saver.aput(config, test_checkpoint, {}, {})

    # Retrieve checkpoint
    retrieved = await saver.aget_tuple(config)
    assert retrieved is not None
    assert retrieved.checkpoint["channel_values"]["workflow_id"] == "wf_test_1"
    assert retrieved.checkpoint["channel_values"]["workflow_status"] == "executing"

    await manager.close()


@pytest.mark.asyncio
async def test_postgres_checkpoint_state_persistence_across_graph_instances() -> None:
    """Verify that a newly instantiated graph loads checkpoints from the same database."""
    alive = await check_postgres_alive()
    if not alive:
        pytest.skip("PostgreSQL test database not available.")

    pool = AsyncConnectionPool(
        POSTGRES_TEST_URL,
        open=False,
        min_size=1,
        max_size=3,
        kwargs={"autocommit": True, "row_factory": dict_row},
    )
    await pool.open(wait=True)

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    saver = AsyncPostgresSaver(conn=pool)
    await saver.setup()

    tid = f"th_persist_{uuid.uuid4().hex[:8]}"

    # Graph 1 runs workflow
    llm1 = MockChatModel(responses={"default": "Mock aggregation response"})
    graph1 = create_agent_graph(llm=llm1, checkpointer=saver)

    # Directly run a custom input
    init_state = {
        "workflow_id": "wf_persist_1",
        "thread_id": tid,
        "workflow_status": WorkflowStatus.PENDING,
        "user_request": "retrieve system status",
        "completed_steps": ["step_0"],
        "pending_steps": [],
        "execution_plan": ExecutionPlan(
            workflow_id="wf_persist_1",
            goal="test",
            user_request="retrieve system status",
            steps=[PlanStep(step_id="step_0", tool_name="health_check", description="test")],
        ),
    }
    config = {"configurable": {"thread_id": tid}}
    await graph1.ainvoke(init_state, config=config)

    # Graph 2 is instantiated from scratch with separate chat model and separate stategraph
    llm2 = MockChatModel(responses={"default": "Graph 2 response"})
    graph2 = create_agent_graph(llm=llm2, checkpointer=saver)

    cp_tuple = await graph2.checkpointer.aget_tuple(config)
    assert cp_tuple is not None
    values = cp_tuple.checkpoint["channel_values"]
    assert values["workflow_id"] == "wf_persist_1"
    assert "step_0" in values["completed_steps"]

    await pool.close()


@pytest.mark.asyncio
async def test_postgres_checkpoint_thread_isolation() -> None:
    """Verify that multiple threads maintain strictly isolated states in PostgreSQL."""
    alive = await check_postgres_alive()
    if not alive:
        pytest.skip("PostgreSQL test database not available.")

    pool = AsyncConnectionPool(
        POSTGRES_TEST_URL,
        open=False,
        min_size=1,
        max_size=3,
        kwargs={"autocommit": True, "row_factory": dict_row},
    )
    await pool.open(wait=True)

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    saver = AsyncPostgresSaver(conn=pool)
    await saver.setup()

    tid_a = f"th_iso_a_{uuid.uuid4().hex[:8]}"
    tid_b = f"th_iso_b_{uuid.uuid4().hex[:8]}"

    config_a = {"configurable": {"thread_id": tid_a, "checkpoint_ns": ""}}
    config_b = {"configurable": {"thread_id": tid_b, "checkpoint_ns": ""}}

    cp_a = {
        "v": 1,
        "ts": "2026-11-08T12:00:00.000000+00:00",
        "id": str(uuid.uuid4()),
        "channel_values": {"thread_id": tid_a, "user_id": "alice"},
        "channel_versions": {"thread_id": 1, "user_id": 1},
        "versions_seen": {},
    }
    cp_b = {
        "v": 1,
        "ts": "2026-11-08T12:00:00.000000+00:00",
        "id": str(uuid.uuid4()),
        "channel_values": {"thread_id": tid_b, "user_id": "bob"},
        "channel_versions": {"thread_id": 1, "user_id": 1},
        "versions_seen": {},
    }

    await saver.aput(config_a, cp_a, {}, {})
    await saver.aput(config_b, cp_b, {}, {})

    res_a = await saver.aget_tuple(config_a)
    res_b = await saver.aget_tuple(config_b)

    assert res_a is not None and res_a.checkpoint["channel_values"]["user_id"] == "alice"
    assert res_b is not None and res_b.checkpoint["channel_values"]["user_id"] == "bob"

    # Missing thread returns None
    config_missing = {"configurable": {"thread_id": "th_non_existent_999"}}
    assert await saver.aget_tuple(config_missing) is None

    await pool.close()


@pytest.mark.asyncio
async def test_postgres_checkpointer_factory_fail_closed() -> None:
    """Verify that PostgreSQL checkpointer fails closed on unreachable host."""
    unreachable_url = (
        "postgresql://invalid_user:invalid_pass@127.0.0.1:54329/invalid_db?connect_timeout=1"
    )

    with pytest.raises(CheckpointerConnectionError):
        await get_checkpointer("postgres", database_url=unreachable_url, timeout=1.0)
