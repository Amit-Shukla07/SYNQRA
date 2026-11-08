"""Durable LangGraph Checkpointer Architecture for SYNQRA (Phase 9.5.2).

Provides factory, connection pooling, and lifecycle management for LangGraph state
persistence across application restarts using PostgreSQL (AsyncPostgresSaver) or
in-memory MemorySaver for isolated development and fast unit testing.
"""

from __future__ import annotations

import contextlib
import logging
import re
from typing import Any
from urllib.parse import urlparse, urlunparse

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from synqra.agents.exceptions import (
    CheckpointerConfigurationError,
    CheckpointerConnectionError,
)
from synqra.core.config import get_settings

logger = logging.getLogger("synqra.agents.checkpointer")


def normalize_postgres_url(raw_url: str) -> str:
    """Normalize SQLAlchemy-formatted PostgreSQL URLs into standard libpq/psycopg URIs.

    Converts drivers like 'postgresql+psycopg://' or 'postgresql+asyncpg://' into
    'postgresql://' compatible with psycopg 3 and psycopg_pool.
    """
    if not raw_url or not isinstance(raw_url, str):
        raise CheckpointerConfigurationError(
            "PostgreSQL connection URL must be a non-empty string."
        )

    cleaned = raw_url.strip()
    if cleaned.startswith("sqlite"):
        raise CheckpointerConfigurationError(
            "SQLite database URL is not supported by PostgreSQL checkpointer. "
            "Configure SYNQRA_DATABASE_URL with a valid PostgreSQL connection string."
        )

    # Replace driver prefixes
    for prefix in ("postgresql+psycopg://", "postgresql+psycopg2://", "postgresql+asyncpg://"):
        if cleaned.startswith(prefix):
            return "postgresql://" + cleaned[len(prefix) :]

    return cleaned


def mask_database_url(url: str) -> str:
    """Safely mask password in a database URL for log messages."""
    try:
        parsed = urlparse(url)
        if parsed.password:
            netloc = parsed.netloc.replace(f":{parsed.password}@", ":***@")
            return urlunparse(parsed._replace(netloc=netloc))
        return url
    except Exception:
        # Fallback regex mask
        return re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", url)


class PostgresCheckpointerManager:
    """Manages the connection pool and lifecycle of AsyncPostgresSaver."""

    def __init__(self) -> None:
        self._pool: Any | None = None
        self._saver: Any | None = None
        self._conn_string: str | None = None

    @property
    def is_initialized(self) -> bool:
        """Return True if pool and saver have been initialized."""
        return self._pool is not None and self._saver is not None

    async def initialize(
        self,
        database_url: str | None = None,
        min_size: int | None = None,
        max_size: int | None = None,
        timeout: float | None = None,
    ) -> BaseCheckpointSaver:
        """Initialize connection pool, run LangGraph schema setup, and return checkpointer.

        Args:
            database_url: Raw database connection string (defaults to settings.database_url).
            min_size: Minimum connection pool size.
            max_size: Maximum connection pool size.
            timeout: Connection acquisition timeout in seconds.

        Returns:
            Configured and initialized AsyncPostgresSaver.

        Raises:
            CheckpointerConfigurationError: If configuration is invalid.
            CheckpointerConnectionError: If connection or schema setup fails.
        """
        if self._saver is not None:
            return self._saver

        cfg = get_settings()
        raw_url = database_url or cfg.database_url
        norm_url = normalize_postgres_url(raw_url)
        self._conn_string = norm_url

        pool_min = min_size if min_size is not None else cfg.checkpointer_pool_min_size
        pool_max = max_size if max_size is not None else cfg.checkpointer_pool_max_size
        pool_timeout = timeout if timeout is not None else cfg.database_pool_timeout

        masked_url = mask_database_url(norm_url)
        logger.info(
            "Initializing PostgreSQL checkpointer (pool min: %d, max: %d, host: %s)",
            pool_min,
            pool_max,
            masked_url,
        )

        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            from psycopg.rows import dict_row
            from psycopg_pool import AsyncConnectionPool
        except ImportError as exc:
            raise CheckpointerConfigurationError(
                f"Missing required PostgreSQL checkpointer dependencies: {exc}. "
                "Ensure 'langgraph-checkpoint-postgres' and 'psycopg-pool' are installed."
            ) from exc

        try:
            pool = AsyncConnectionPool(
                norm_url,
                open=False,
                min_size=pool_min,
                max_size=pool_max,
                timeout=pool_timeout,
                kwargs={"autocommit": True, "row_factory": dict_row},
            )
            await pool.open(wait=True, timeout=pool_timeout)
            self._pool = pool

            saver = AsyncPostgresSaver(conn=pool)
            # Ensure LangGraph checkpoint tables (checkpoints, checkpoint_blobs, etc.) exist
            await saver.setup()
            self._saver = saver
            logger.info("PostgreSQL checkpointer initialized and schema verified successfully.")
            return saver
        except Exception as exc:
            # Cleanup partially opened pool on failure
            if self._pool is not None:
                with contextlib.suppress(Exception):
                    await self._pool.close()
                self._pool = None
            self._saver = None
            raise CheckpointerConnectionError(
                f"Failed to connect to PostgreSQL checkpointer at '{masked_url}': {exc}"
            ) from exc

    async def get_checkpointer(self) -> BaseCheckpointSaver:
        """Return initialized checkpointer or initialize with defaults."""
        if self._saver is not None:
            return self._saver
        return await self.initialize()

    async def check_health(self) -> dict[str, Any]:
        """Perform a diagnostic ping on the checkpointer database connection pool."""
        if self._pool is None:
            return {
                "status": "uninitialized",
                "backend": "postgres",
                "healthy": False,
                "error": "PostgresCheckpointerManager is not initialized.",
            }

        try:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute("SELECT 1")
                row = await cur.fetchone()
                if row and row["?column?"] == 1:
                    return {
                        "status": "connected",
                        "backend": "postgres",
                        "healthy": True,
                    }
            return {
                "status": "degraded",
                "backend": "postgres",
                "healthy": False,
                "error": "Unexpected probe response.",
            }
        except Exception as exc:
            return {
                "status": "error",
                "backend": "postgres",
                "healthy": False,
                "error": str(exc),
            }

    async def close(self) -> None:
        """Gracefully drain and close the connection pool."""
        if self._pool is not None:
            try:
                await self._pool.close()
                logger.info("PostgreSQL checkpointer connection pool closed gracefully.")
            except Exception as exc:
                logger.warning("Error closing PostgreSQL checkpointer pool: %s", exc)
            finally:
                self._pool = None
                self._saver = None


# Singleton manager instance
_global_postgres_manager: PostgresCheckpointerManager | None = None


def get_postgres_checkpointer_manager() -> PostgresCheckpointerManager:
    """Return the singleton PostgresCheckpointerManager instance."""
    global _global_postgres_manager
    if _global_postgres_manager is None:
        _global_postgres_manager = PostgresCheckpointerManager()
    return _global_postgres_manager


async def get_checkpointer(
    checkpointer_type: str | None = None,
    database_url: str | None = None,
    timeout: float | None = None,
) -> BaseCheckpointSaver:
    """Checkpointer factory resolving between in-memory and PostgreSQL persistence.

    Args:
        checkpointer_type: Explicit checkpointer type ('memory' or 'postgres').
                           Defaults to SYNQRA_CHECKPOINTER_TYPE setting.
        database_url: Optional database connection URL when type is 'postgres'.
        timeout: Optional timeout in seconds for PostgreSQL connection acquisition.

    Returns:
        BaseCheckpointSaver instance (MemorySaver or AsyncPostgresSaver).

    Raises:
        CheckpointerConfigurationError: When an unknown type is configured or config is invalid.
        CheckpointerConnectionError: When PostgreSQL is requested but connection fails.
    """
    resolved_type = (checkpointer_type or get_settings().checkpointer_type).lower().strip()

    if resolved_type == "memory":
        return MemorySaver()

    if resolved_type == "postgres":
        manager = get_postgres_checkpointer_manager()
        return await manager.initialize(database_url=database_url, timeout=timeout)

    raise CheckpointerConfigurationError(
        f"Unsupported checkpointer type '{resolved_type}'. "
        "Supported backends are 'memory' and 'postgres'."
    )


async def close_checkpointer() -> None:
    """Shutdown hook to close active checkpointer connection pools."""
    global _global_postgres_manager
    if _global_postgres_manager is not None:
        await _global_postgres_manager.close()
        _global_postgres_manager = None
