"""Standardized Pydantic data models for SYNQRA service integrations.

Defines lifecycle states, metadata descriptions, and operational health
reporting for all external service connectors.
"""

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    """Return the current timezone-aware UTC datetime."""
    return datetime.now(UTC)


class IntegrationState(StrEnum):
    """Operational state of an integration."""

    UNCONFIGURED = "unconfigured"
    CONFIGURED = "configured"
    CONNECTED = "connected"
    DEGRADED = "degraded"
    ERROR = "error"


class IntegrationMetadata(BaseModel):
    """Static metadata describing an integration and its capabilities."""

    name: str = Field(
        description="Unique slug identifier for the integration (e.g. 'youtube', 'github')."
    )
    display_name: str = Field(description="Human-readable integration name.")
    description: str = Field(description="Overview of capabilities and exposed tools.")
    version: str = Field(default="0.1.0", description="Integration connector version.")
    auth_required: bool = Field(
        default=True,
        description="Whether this integration requires authentication credentials.",
    )
    auth_type: str = Field(
        default="api_key",
        description="Authentication mechanism ('api_key', 'oauth2', 'composio', 'none').",
    )
    documentation_url: str | None = Field(
        default=None, description="Official documentation or setup URL."
    )


class IntegrationStatus(BaseModel):
    """Dynamic operational health and connection status for an integration."""

    name: str = Field(description="Unique slug identifier of the integration.")
    display_name: str = Field(description="Human-readable integration name.")
    state: IntegrationState = Field(
        default=IntegrationState.UNCONFIGURED,
        description="Current operational lifecycle state.",
    )
    is_configured: bool = Field(
        default=False,
        description="Whether required configuration or credentials are present.",
    )
    is_connected: bool = Field(
        default=False,
        description="Whether the integration successfully established connection to the service.",
    )
    registered_tools: list[str] = Field(
        default_factory=list,
        description="List of tool names currently registered with the FastMCP server.",
    )
    error_message: str | None = Field(
        default=None, description="Diagnostic error details if state is DEGRADED or ERROR."
    )
    last_checked: datetime = Field(
        default_factory=utc_now,
        description="Timestamp of the most recent health/connectivity check.",
    )
