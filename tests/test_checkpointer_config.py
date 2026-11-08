"""Tests for standardized checkpointer configuration (Phase 9.5.2).

Verifies canonical environment variable (SYNQRA_CHECKPOINTER_TYPE),
rejection of invalid modes, failure on unreachable PostgreSQL,
never falling back to memory in postgres mode, and handling of legacy variables.
"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import MemorySaver

from synqra.agents.checkpointer import (
    close_checkpointer,
    get_checkpointer,
)
from synqra.agents.exceptions import (
    CheckpointerConfigurationError,
    CheckpointerConnectionError,
)
from synqra.core.config import Settings
from synqra.server.app import run_server

POSTGRES_TEST_URL = "postgresql://amitshukla@localhost:5432/synqra_test"
UNREACHABLE_POSTGRES_URL = "postgresql://invalid_user:invalid_pass@127.0.0.1:59999/nonexistent_db"


@pytest.mark.asyncio
async def test_memory_mode_selects_memory_saver() -> None:
    """Verify that SYNQRA_CHECKPOINTER_TYPE=memory selects MemorySaver."""
    saver = await get_checkpointer(checkpointer_type="memory")
    assert isinstance(saver, MemorySaver)


@pytest.mark.asyncio
async def test_postgres_mode_selects_async_postgres_saver() -> None:
    """Verify that SYNQRA_CHECKPOINTER_TYPE=postgres selects AsyncPostgresSaver."""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    try:
        saver = await get_checkpointer(checkpointer_type="postgres", database_url=POSTGRES_TEST_URL)
        assert isinstance(saver, AsyncPostgresSaver)
    finally:
        await close_checkpointer()


@pytest.mark.asyncio
async def test_invalid_mode_is_rejected() -> None:
    """Verify that unsupported checkpointer backends raise a clear configuration error."""
    # 1. Direct factory rejection
    with pytest.raises(CheckpointerConfigurationError, match="Unsupported checkpointer type"):
        await get_checkpointer(checkpointer_type="redis")

    with pytest.raises(CheckpointerConfigurationError, match="Unsupported checkpointer type"):
        await get_checkpointer(checkpointer_type="sqlite")

    # 2. Pydantic settings rejection
    with pytest.raises(ValueError, match="Invalid checkpointer type"):
        Settings(checkpointer_type="mongodb")


@pytest.mark.asyncio
async def test_postgres_mode_never_falls_back_to_memory() -> None:
    """Verify that if PostgreSQL connection fails, it raises CheckpointerConnectionError
    and NEVER falls back to MemorySaver.
    """
    with pytest.raises(CheckpointerConnectionError, match="Failed to connect to PostgreSQL"):
        await get_checkpointer(
            checkpointer_type="postgres",
            database_url=UNREACHABLE_POSTGRES_URL,
            timeout=1.0,
        )


def test_postgresql_connection_failure_prevents_server_startup() -> None:
    """Verify that unreachable PostgreSQL causes server startup pre-flight to fail closed."""
    failing_settings = Settings(
        checkpointer_type="postgres",
        database_url=UNREACHABLE_POSTGRES_URL,
    )

    with pytest.raises(CheckpointerConnectionError):
        run_server(settings=failing_settings)


def test_canonical_environment_variable_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that SYNQRA_CHECKPOINTER_TYPE environment variable is respected."""
    monkeypatch.setenv("SYNQRA_CHECKPOINTER_TYPE", "postgres")
    cfg = Settings()
    assert cfg.checkpointer_type == "postgres"

    monkeypatch.setenv("SYNQRA_CHECKPOINTER_TYPE", "memory")
    cfg2 = Settings()
    assert cfg2.checkpointer_type == "memory"


def test_legacy_agent_checkpointer_compatibility_and_deprecation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify legacy SYNQRA_AGENT_CHECKPOINTER produces a deprecation warning."""
    monkeypatch.delenv("SYNQRA_CHECKPOINTER_TYPE", raising=False)
    monkeypatch.setenv("SYNQRA_AGENT_CHECKPOINTER", "postgres")

    with pytest.deprecated_call(match="SYNQRA_AGENT_CHECKPOINTER is deprecated"):
        cfg = Settings()
        assert cfg.checkpointer_type == "postgres"


def test_conflicting_checkpointer_variables_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that specifying conflicting canonical and legacy variables raises ValueError."""
    monkeypatch.setenv("SYNQRA_CHECKPOINTER_TYPE", "postgres")
    monkeypatch.setenv("SYNQRA_AGENT_CHECKPOINTER", "memory")

    with pytest.raises(ValueError, match="Conflicting checkpointer configuration"):
        Settings()
