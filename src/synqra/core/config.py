"""Configuration management for SYNQRA using Pydantic Settings.

Provides strongly-typed, environment-aware configuration loaded securely
from environment variables and optional .env files.
"""

import os
import warnings
from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings for SYNQRA MCP server."""

    app_name: str = Field(
        default="SYNQRA",
        validation_alias=AliasChoices("SYNQRA_APP_NAME", "APP_NAME"),
        description="Application name displayed across MCP and logs.",
    )
    app_version: str = Field(
        default="0.1.0",
        validation_alias=AliasChoices("SYNQRA_APP_VERSION", "APP_VERSION"),
        description="Semantic version of the SYNQRA application.",
    )
    host: str = Field(
        default="127.0.0.1",
        validation_alias=AliasChoices("SYNQRA_HOST", "HOST"),
        description="Host interface to bind the MCP server.",
    )
    port: int = Field(
        default=8000,
        validation_alias=AliasChoices("SYNQRA_PORT", "PORT"),
        description="Port number to bind the MCP server.",
    )
    debug: bool = Field(
        default=False,
        validation_alias=AliasChoices("SYNQRA_DEBUG", "DEBUG"),
        description="Flag to enable debug mode and verbose logging.",
    )
    environment: Literal["development", "staging", "production", "testing"] = Field(
        default="development",
        validation_alias=AliasChoices("SYNQRA_ENVIRONMENT", "ENVIRONMENT", "ENV"),
        description="Runtime environment identifier.",
    )
    transport: Literal["streamable-http", "stdio", "sse", "http"] = Field(
        default="streamable-http",
        validation_alias=AliasChoices("SYNQRA_TRANSPORT", "TRANSPORT"),
        description="MCP transport mechanism.",
    )
    mcp_path: str = Field(
        default="/mcp",
        validation_alias=AliasChoices("SYNQRA_MCP_PATH", "MCP_PATH"),
        description="URL path mount for streamable-http transport.",
    )
    stateless_http: bool = Field(
        default=True,
        validation_alias=AliasChoices("SYNQRA_STATELESS_HTTP", "STATELESS_HTTP"),
        description=(
            "Allow stateless HTTP requests without requiring an mcp-session-id header. "
            "Essential for compatibility with Cursor and direct HTTP clients."
        ),
    )
    database_url: str = Field(
        default="sqlite+aiosqlite:///./synqra.db",
        validation_alias=AliasChoices("SYNQRA_DATABASE_URL", "DATABASE_URL"),
        description="Async database connection string (PostgreSQL psycopg or SQLite aiosqlite).",
    )
    database_pool_size: int = Field(
        default=5,
        validation_alias=AliasChoices("SYNQRA_DATABASE_POOL_SIZE", "DATABASE_POOL_SIZE"),
        description="PostgreSQL async connection pool size.",
    )
    database_max_overflow: int = Field(
        default=10,
        validation_alias=AliasChoices("SYNQRA_DATABASE_MAX_OVERFLOW", "DATABASE_MAX_OVERFLOW"),
        description="Maximum temporary connections above pool_size in PostgreSQL.",
    )
    database_pool_timeout: float = Field(
        default=30.0,
        validation_alias=AliasChoices("SYNQRA_DATABASE_POOL_TIMEOUT", "DATABASE_POOL_TIMEOUT"),
        description="Seconds to wait before timing out on acquiring a pool connection.",
    )
    database_pool_recycle: int = Field(
        default=1800,
        validation_alias=AliasChoices("SYNQRA_DATABASE_POOL_RECYCLE", "DATABASE_POOL_RECYCLE"),
        description="Seconds after which connections are recycled (1800s = 30min).",
    )
    database_pool_pre_ping: bool = Field(
        default=True,
        validation_alias=AliasChoices("SYNQRA_DATABASE_POOL_PRE_PING", "DATABASE_POOL_PRE_PING"),
        description="Emit a test ping on connection checkout to discard dead connections.",
    )
    database_echo: bool = Field(
        default=False,
        validation_alias=AliasChoices("SYNQRA_DATABASE_ECHO", "DATABASE_ECHO"),
        description="Log raw SQL statements. Kept False to avoid leaking sensitive query params.",
    )
    composio_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("SYNQRA_COMPOSIO_API_KEY", "COMPOSIO_API_KEY"),
        description="Composio API key for Google Drive, Calendar, and Gmail integrations.",
    )
    composio_user_id: str = Field(
        default="synqra_default_user",
        validation_alias=AliasChoices("SYNQRA_COMPOSIO_USER_ID", "COMPOSIO_USER_ID"),
        description="Stable user identifier for scoping Composio sessions.",
    )
    youtube_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("SYNQRA_YOUTUBE_API_KEY", "YOUTUBE_API_KEY"),
        description="Google Cloud API key for YouTube Data API v3.",
    )
    github_token: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("SYNQRA_GITHUB_TOKEN", "GITHUB_TOKEN", "GITHUB_PAT"),
        description="GitHub Personal Access Token for GitHub API access.",
    )
    tavily_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("SYNQRA_TAVILY_API_KEY", "TAVILY_API_KEY"),
        description="API key for Tavily Web Search and Content Extraction.",
    )
    llm_provider: str = Field(
        default="openai",
        validation_alias=AliasChoices("SYNQRA_LLM_PROVIDER", "LLM_PROVIDER"),
        description="Primary LLM provider for agentic planning (e.g. openai, anthropic, google).",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices("SYNQRA_LLM_MODEL", "LLM_MODEL"),
        description="Model name for reasoning and planning.",
    )
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("SYNQRA_OPENAI_API_KEY", "OPENAI_API_KEY"),
        description="OpenAI API key for LLM reasoning and agent orchestration.",
    )
    approval_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("SYNQRA_APPROVAL_API_KEY", "APPROVAL_API_KEY"),
        description="High-entropy master bearer token for human approval API authentication.",
    )
    approval_tokens: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("SYNQRA_APPROVAL_TOKENS", "APPROVAL_TOKENS"),
        description=(
            "Configured token-to-operator mapping (e.g. 'token1:alice:admin,token2:bob:operator' "
            "or JSON string). Used for multi-operator environments."
        ),
    )
    approval_default_operator: str = Field(
        default="primary_operator",
        validation_alias=AliasChoices(
            "SYNQRA_APPROVAL_DEFAULT_OPERATOR", "APPROVAL_DEFAULT_OPERATOR"
        ),
        description="Operator username bound to the primary approval API key.",
    )
    approval_default_roles: str = Field(
        default="operator",
        validation_alias=AliasChoices(
            "SYNQRA_APPROVAL_DEFAULT_ROLES",
            "APPROVAL_DEFAULT_ROLES",
            "SYNQRA_APPROVAL_DEFAULT_ROLE",
        ),
        description="Explicit roles granted to primary API key (default: 'operator').",
    )
    approval_default_allowed_workflows: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "SYNQRA_APPROVAL_DEFAULT_ALLOWED_WORKFLOWS",
            "APPROVAL_DEFAULT_ALLOWED_WORKFLOWS",
        ),
        description="Comma-separated workflow IDs permitted for primary key.",
    )
    approval_secret_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("SYNQRA_APPROVAL_SECRET_KEY", "APPROVAL_SECRET_KEY"),
        description="High-entropy secret key used for signing approval binding tokens.",
    )
    approval_store_type: str = Field(
        default="memory",
        validation_alias=AliasChoices("SYNQRA_APPROVAL_STORE_TYPE", "APPROVAL_STORE_TYPE"),
        description="Approval storage backend ('memory' or 'database').",
    )
    checkpointer_type: str = Field(
        default="memory",
        validation_alias=AliasChoices("SYNQRA_CHECKPOINTER_TYPE", "CHECKPOINTER_TYPE"),
        description="LangGraph checkpoint storage backend ('memory' or 'postgres').",
    )
    checkpointer_pool_min_size: int = Field(
        default=1,
        validation_alias=AliasChoices(
            "SYNQRA_CHECKPOINTER_POOL_MIN_SIZE", "CHECKPOINTER_POOL_MIN_SIZE"
        ),
        description="Minimum connection pool size for PostgreSQL checkpointer.",
    )
    checkpointer_pool_max_size: int = Field(
        default=5,
        validation_alias=AliasChoices(
            "SYNQRA_CHECKPOINTER_POOL_MAX_SIZE", "CHECKPOINTER_POOL_MAX_SIZE"
        ),
        description="Maximum connection pool size for PostgreSQL checkpointer.",
    )

    @field_validator("checkpointer_type")
    @classmethod
    def validate_checkpointer_type(cls, v: str) -> str:
        """Resolve canonical SYNQRA_CHECKPOINTER_TYPE and legacy SYNQRA_AGENT_CHECKPOINTER."""
        canonical_env = os.environ.get("SYNQRA_CHECKPOINTER_TYPE") or os.environ.get(
            "CHECKPOINTER_TYPE"
        )
        legacy_env = os.environ.get("SYNQRA_AGENT_CHECKPOINTER") or os.environ.get(
            "AGENT_CHECKPOINTER"
        )

        if canonical_env and legacy_env:
            c_clean = canonical_env.strip().lower()
            l_clean = legacy_env.strip().lower()
            if c_clean != l_clean:
                raise ValueError(
                    f"Conflicting checkpointer configuration: canonical "
                    f"SYNQRA_CHECKPOINTER_TYPE='{canonical_env}' conflicts with legacy "
                    f"SYNQRA_AGENT_CHECKPOINTER='{legacy_env}'. Use only SYNQRA_CHECKPOINTER_TYPE."
                )
            warnings.warn(
                "SYNQRA_AGENT_CHECKPOINTER is deprecated; use SYNQRA_CHECKPOINTER_TYPE instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            val = c_clean
        elif legacy_env and not canonical_env:
            warnings.warn(
                "SYNQRA_AGENT_CHECKPOINTER is deprecated; use SYNQRA_CHECKPOINTER_TYPE instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            val = legacy_env.strip().lower()
        else:
            val = v.strip().lower()

        if val not in ("memory", "postgres"):
            raise ValueError(
                f"Invalid checkpointer type '{v}'. Supported backends are 'memory' and 'postgres'."
            )
        return val

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton instance of application settings."""
    return Settings()
