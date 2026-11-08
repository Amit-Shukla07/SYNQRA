"""Tests for SQLAlchemy async database models and session management."""

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from synqra.database.models import AuditLog, IntegrationCredential
from synqra.database.session import init_db


@pytest.mark.asyncio
async def test_database_initialization_and_crud() -> None:
    """Verify table creation, record insertion, and queries using SQLite in-memory."""
    test_engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    await init_db(test_engine)

    session_factory = async_sessionmaker(
        bind=test_engine,
        expire_on_commit=False,
    )

    async with session_factory() as session:
        # 1. Insert IntegrationCredential
        cred = IntegrationCredential(
            integration_name="github",
            is_enabled=True,
            auth_type="oauth2",
            account_identifier="synqra-bot",
            extra_metadata={"scope": "repo,read:user"},
        )
        session.add(cred)

        # 2. Insert AuditLog
        log_entry = AuditLog(
            session_id="session-1234",
            tool_name="web_search",
            status="success",
            execution_time_ms=124.5,
        )
        session.add(log_entry)
        await session.commit()

    async with session_factory() as session:
        # Query credential back
        stmt_cred = select(IntegrationCredential).where(
            IntegrationCredential.integration_name == "github"
        )
        result_cred = await session.scalar(stmt_cred)
        assert result_cred is not None
        assert result_cred.integration_name == "github"
        assert result_cred.is_enabled is True
        assert result_cred.extra_metadata == {"scope": "repo,read:user"}
        assert result_cred.created_at is not None

        # Query audit log back
        stmt_log = select(AuditLog).where(AuditLog.session_id == "session-1234")
        result_log = await session.scalar(stmt_log)
        assert result_log is not None
        assert result_log.tool_name == "web_search"
        assert result_log.status == "success"
        assert result_log.execution_time_ms == 124.5

    await test_engine.dispose()
