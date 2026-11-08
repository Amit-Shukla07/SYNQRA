"""Database session management and async engine configuration for SYNQRA."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from synqra.core.config import get_settings
from synqra.database.base import Base

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_async_engine(database_url: str | None = None) -> AsyncEngine:
    """Return a singleton async SQLAlchemy engine with connection pooling and security controls."""
    global _engine
    if _engine is None or database_url is not None:
        cfg = get_settings()
        url = database_url or cfg.database_url

        kwargs: dict[str, object] = {
            "echo": cfg.database_echo,
            "future": True,
        }
        # Connection pooling configuration for PostgreSQL/MySQL
        # (SQLite does not accept queue pool parameters)
        if not url.startswith("sqlite"):
            kwargs.update(
                {
                    "pool_size": cfg.database_pool_size,
                    "max_overflow": cfg.database_max_overflow,
                    "pool_timeout": cfg.database_pool_timeout,
                    "pool_recycle": cfg.database_pool_recycle,
                    "pool_pre_ping": cfg.database_pool_pre_ping,
                }
            )

        engine = create_async_engine(url, **kwargs)
        if database_url is None:
            _engine = engine
        return engine
    return _engine


def get_async_session_maker(
    engine: AsyncEngine | None = None,
) -> async_sessionmaker[AsyncSession]:
    """Return an async sessionmaker instance."""
    global _session_factory
    eng = engine or get_async_engine()
    if _session_factory is None or engine is not None:
        factory = async_sessionmaker(
            bind=eng,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
        if engine is None:
            _session_factory = factory
        return factory
    return _session_factory


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide an asynchronous transactional database session scope."""
    factory = get_async_session_maker()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db(engine: AsyncEngine | None = None) -> None:
    """Initialize database tables for registered SQLAlchemy models."""
    eng = engine or get_async_engine()
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
