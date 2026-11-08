"""Common base abstraction for all SYNQRA external service integrations.

Defines the contract for credential verification, connectivity probes,
health reporting, and FastMCP tool registration with strict error isolation.
"""

import logging
from abc import ABC, abstractmethod

from fastmcp import FastMCP

from synqra.integrations.exceptions import (
    IntegrationRegistrationError,
    SynqraIntegrationError,
)
from synqra.integrations.models import (
    IntegrationMetadata,
    IntegrationState,
    IntegrationStatus,
    utc_now,
)

logger = logging.getLogger("synqra.integrations.base")


class BaseIntegration(ABC):
    """Abstract base class for all external service integrations in SYNQRA.

    Subclasses implement service-specific authentication checks, connection tests,
    and FastMCP tool registrations while inheriting standardized status reporting
    and failure isolation.
    """

    def __init__(self, metadata: IntegrationMetadata) -> None:
        self._metadata = metadata
        self._registered_tools: list[str] = []

    @property
    def metadata(self) -> IntegrationMetadata:
        """Integration metadata including name, display name, and auth requirements."""
        return self._metadata

    @property
    def name(self) -> str:
        """Unique slug identifier for the integration."""
        return self._metadata.name

    @property
    def display_name(self) -> str:
        """Human-readable name of the integration."""
        return self._metadata.display_name

    @property
    def description(self) -> str:
        """Description of capabilities provided by this integration."""
        return self._metadata.description

    @property
    def registered_tools(self) -> list[str]:
        """List of tools successfully registered on the MCP server."""
        return list(self._registered_tools)

    @abstractmethod
    def is_configured(self) -> bool:
        """Check whether the necessary credentials, keys, or configuration are present.

        Returns:
            True if all required configuration is present, False otherwise.
        """
        ...

    @abstractmethod
    async def test_connection(self) -> bool:
        """Perform a live connectivity or authentication check against the upstream service.

        Returns:
            True if connected and operational, False if unreachable or unauthenticated.
        """
        ...

    @abstractmethod
    def _register_tools_impl(self, server: FastMCP) -> list[str]:
        """Internal implementation for registering integration-specific tools on FastMCP.

        Args:
            server: The FastMCP server instance.

        Returns:
            A list of registered tool names.
        """
        ...

    def register_tools(self, server: FastMCP) -> list[str]:
        """Safely register integration tools with FastMCP, isolating any registration failures.

        Args:
            server: The FastMCP server instance.

        Returns:
            List of registered tool names. Returns empty list if unconfigured or on failure.
        """
        if not self.is_configured():
            logger.debug(
                "Skipping tool registration for unconfigured integration: %s",
                self.name,
            )
            self._registered_tools = []
            return []

        try:
            tools = self._register_tools_impl(server)
            self._registered_tools = tools
            logger.info(
                "Registered %d tool(s) for integration '%s': %s",
                len(tools),
                self.name,
                ", ".join(tools) if tools else "none",
            )
            return tools
        except Exception as exc:
            logger.error(
                "Failed to register tools for integration '%s': %s",
                self.name,
                exc,
                exc_info=True,
            )
            self._registered_tools = []
            raise IntegrationRegistrationError(
                f"Tool registration failed: {exc}",
                integration_name=self.name,
            ) from exc

    async def health_check(self) -> IntegrationStatus:
        """Perform a standardized health check and return detailed operational status.

        Follows a resilient template method:
        1. Checks configuration readiness via is_configured().
        2. Probes live connectivity via test_connection() inside an isolated try/except block.
        3. Returns an IntegrationStatus model with honest state reporting.
        """
        configured = False
        try:
            configured = self.is_configured()
        except Exception as exc:
            logger.warning("Error checking configuration for '%s': %s", self.name, exc)
            return IntegrationStatus(
                name=self.name,
                display_name=self.display_name,
                state=IntegrationState.ERROR,
                is_configured=False,
                is_connected=False,
                registered_tools=self._registered_tools,
                error_message=f"Configuration error: {exc}",
                last_checked=utc_now(),
            )

        if not configured:
            return IntegrationStatus(
                name=self.name,
                display_name=self.display_name,
                state=IntegrationState.UNCONFIGURED,
                is_configured=False,
                is_connected=False,
                registered_tools=self._registered_tools,
                error_message=None,
                last_checked=utc_now(),
            )

        # Configured: probe connectivity with error isolation
        try:
            connected = await self.test_connection()
            if connected:
                state = IntegrationState.CONNECTED
                error_msg = None
            else:
                state = IntegrationState.DEGRADED
                error_msg = "Upstream service connection test returned False"
        except SynqraIntegrationError as exc:
            logger.warning("Integration connection failed for '%s': %s", self.name, exc)
            state = IntegrationState.ERROR
            connected = False
            error_msg = str(exc)
        except Exception as exc:
            logger.error("Unexpected error testing connection for '%s': %s", self.name, exc)
            state = IntegrationState.ERROR
            connected = False
            error_msg = f"Unexpected connection error: {exc}"

        return IntegrationStatus(
            name=self.name,
            display_name=self.display_name,
            state=state,
            is_configured=True,
            is_connected=connected,
            registered_tools=self._registered_tools,
            error_message=error_msg,
            last_checked=utc_now(),
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name!r}, configured={self.is_configured()})>"
