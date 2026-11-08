"""Central integration registry for SYNQRA.

Manages the registration, retrieval, tool mounting, and health checks for all
external service integrations with complete failure isolation.
"""

import logging
from collections.abc import Sequence

from fastmcp import FastMCP

from synqra.integrations.base import BaseIntegration
from synqra.integrations.exceptions import IntegrationNotFoundError
from synqra.integrations.models import IntegrationStatus

logger = logging.getLogger("synqra.integrations.registry")

ALL_PLANNED_INTEGRATIONS: tuple[str, ...] = (
    "youtube",
    "google_drive",
    "github",
    "google_calendar",
    "web_search",
    "gmail",
)


class IntegrationRegistry:
    """Registry managing external service integrations and their FastMCP tool bindings."""

    def __init__(self, planned: Sequence[str] = ALL_PLANNED_INTEGRATIONS) -> None:
        self._planned_names: list[str] = list(planned)
        self._integrations: dict[str, BaseIntegration] = {}
        self._mounted_tools: dict[str, list[str]] = {}

    def register(self, integration: BaseIntegration) -> None:
        """Register a service integration instance with the registry.

        Args:
            integration: An instance of BaseIntegration.

        Raises:
            TypeError: If integration is not an instance of BaseIntegration.
        """
        if not isinstance(integration, BaseIntegration):
            raise TypeError(f"Expected BaseIntegration instance, got {type(integration).__name__}")

        name = integration.name
        if name in self._integrations:
            logger.warning("Overwriting existing integration registration for: %s", name)

        self._integrations[name] = integration
        logger.debug(
            "Registered integration '%s' (configured=%s)",
            name,
            integration.is_configured(),
        )

    def get(self, name: str) -> BaseIntegration | None:
        """Retrieve an integration by its unique slug identifier, or None if not found."""
        return self._integrations.get(name)

    def get_or_raise(self, name: str) -> BaseIntegration:
        """Retrieve an integration or raise IntegrationNotFoundError if missing."""
        integration = self._integrations.get(name)
        if integration is None:
            raise IntegrationNotFoundError(name)
        return integration

    def list_all(self) -> list[BaseIntegration]:
        """Return a list of all registered integration instances."""
        return list(self._integrations.values())

    def list_names(self) -> list[str]:
        """Return the slugs of all registered integrations."""
        return list(self._integrations.keys())

    def list_available(self) -> list[str]:
        """Return names of all registered integrations that are configured and operational."""
        available: list[str] = []
        for name, integration in self._integrations.items():
            try:
                if integration.is_configured():
                    available.append(name)
            except Exception as exc:
                logger.warning("Error checking configuration for '%s': %s", name, exc)
        return available

    def list_planned(self) -> list[str]:
        """Return names of integrations that are planned but not yet operational."""
        available = set(self.list_available())
        return [name for name in self._planned_names if name not in available]

    def mount_all(self, server: FastMCP) -> dict[str, list[str]]:
        """Mount all configured integrations onto the FastMCP server with failure isolation.

        If one integration fails during registration, the error is isolated and logged,
        allowing other integrations to mount without disruption.

        Args:
            server: The FastMCP server instance.

        Returns:
            Mapping of integration name to list of registered tool names.
        """
        self._mounted_tools.clear()

        for name, integration in self._integrations.items():
            if not integration.is_configured():
                logger.debug("Skipping unconfigured integration: %s", name)
                self._mounted_tools[name] = []
                continue

            try:
                tools = integration.register_tools(server)
                self._mounted_tools[name] = tools
            except Exception as exc:
                logger.error(
                    "Failure mounting integration '%s' (isolated): %s",
                    name,
                    exc,
                    exc_info=True,
                )
                self._mounted_tools[name] = []

        return dict(self._mounted_tools)

    async def check_health_all(self) -> dict[str, IntegrationStatus]:
        """Perform concurrent health checks across all registered integrations.

        Ensures that any failure in one integration is isolated and reported as an
        ERROR state without preventing other integrations from completing.

        Returns:
            Dictionary mapping integration name to its IntegrationStatus.
        """
        statuses: dict[str, IntegrationStatus] = {}

        for name, integration in self._integrations.items():
            try:
                statuses[name] = await integration.health_check()
            except Exception as exc:
                logger.error("Unexpected error in health check for '%s': %s", name, exc)
                from synqra.integrations.models import IntegrationState, utc_now

                statuses[name] = IntegrationStatus(
                    name=name,
                    display_name=integration.display_name,
                    state=IntegrationState.ERROR,
                    is_configured=False,
                    is_connected=False,
                    registered_tools=[],
                    error_message=f"Health check execution error: {exc}",
                    last_checked=utc_now(),
                )

        return statuses


# Default global integration registry
default_registry = IntegrationRegistry()
