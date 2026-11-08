"""Comprehensive tests for SYNQRA Phase 2 Integration Architecture."""

import pytest
from fastmcp import FastMCP

from synqra.integrations.base import BaseIntegration
from synqra.integrations.exceptions import (
    IntegrationConfigurationError,
    IntegrationConnectionError,
    IntegrationNotFoundError,
    SynqraIntegrationError,
)
from synqra.integrations.models import (
    IntegrationMetadata,
    IntegrationState,
    IntegrationStatus,
)
from synqra.integrations.registry import ALL_PLANNED_INTEGRATIONS, IntegrationRegistry
from synqra.server.mcp_server import create_mcp_server


class DummyConfiguredIntegration(BaseIntegration):
    """Stub integration that simulates a successfully configured service."""

    def __init__(self, name: str = "dummy_service") -> None:
        super().__init__(
            metadata=IntegrationMetadata(
                name=name,
                display_name="Dummy Service",
                description="Test service for integration architecture.",
                version="1.0.0",
                auth_required=True,
                auth_type="api_key",
            )
        )

    def is_configured(self) -> bool:
        return True

    async def test_connection(self) -> bool:
        return True

    def _register_tools_impl(self, server: FastMCP) -> list[str]:
        @server.tool(name=f"{self.name}_ping")
        async def ping() -> str:
            return "pong"

        return [f"{self.name}_ping"]


class DummyFailingIntegration(BaseIntegration):
    """Stub integration that simulates connectivity or registration failures."""

    def __init__(self, name: str = "failing_service", fail_on_register: bool = False) -> None:
        super().__init__(
            metadata=IntegrationMetadata(
                name=name,
                display_name="Failing Service",
                description="Simulates errors.",
                version="1.0.0",
            )
        )
        self.fail_on_register = fail_on_register

    def is_configured(self) -> bool:
        return True

    async def test_connection(self) -> bool:
        raise IntegrationConnectionError("Upstream server unreachable", integration_name=self.name)

    def _register_tools_impl(self, _server: FastMCP) -> list[str]:
        if self.fail_on_register:
            raise RuntimeError("Registration explosion")
        return []


def test_integration_status_models() -> None:
    """Verify that IntegrationStatus model behaves correctly."""
    status = IntegrationStatus(
        name="test_service",
        display_name="Test Service",
        state=IntegrationState.CONFIGURED,
        is_configured=True,
        is_connected=False,
    )
    assert status.name == "test_service"
    assert status.state == IntegrationState.CONFIGURED
    assert status.is_connected is False
    assert status.error_message is None
    assert status.last_checked is not None


def test_integration_registry_crud() -> None:
    """Verify registration, retrieval, and error handling for nonexistent integrations."""
    registry = IntegrationRegistry(planned=["service_a", "service_b"])
    integration = DummyConfiguredIntegration(name="service_a")

    registry.register(integration)

    # Retrieval
    assert registry.get("service_a") is integration
    assert registry.get("nonexistent") is None

    with pytest.raises(IntegrationNotFoundError):
        registry.get_or_raise("nonexistent")

    # Type validation
    with pytest.raises(TypeError):
        registry.register("not_an_integration")  # type: ignore[arg-type]


def test_registry_availability_tracking() -> None:
    """Verify available vs planned tracking."""
    registry = IntegrationRegistry(planned=["service_a", "service_b", "service_c"])

    configured = DummyConfiguredIntegration(name="service_a")
    registry.register(configured)

    assert registry.list_available() == ["service_a"]
    assert registry.list_planned() == ["service_b", "service_c"]


def test_failure_isolation_during_tool_mounting() -> None:
    """Requirement 8: Ensure one integration failure does not affect other integrations."""
    registry = IntegrationRegistry()

    good_integration = DummyConfiguredIntegration(name="good_service")
    broken_integration = DummyFailingIntegration(name="broken_service", fail_on_register=True)

    registry.register(broken_integration)
    registry.register(good_integration)

    server = FastMCP("TestIsolationServer")
    mounted = registry.mount_all(server)

    # The broken integration failed gracefully and was recorded with 0 tools
    assert mounted["broken_service"] == []
    # The good integration succeeded without disruption
    assert mounted["good_service"] == ["good_service_ping"]


@pytest.mark.asyncio
async def test_failure_isolation_during_health_check() -> None:
    """Requirement 8: Ensure connection failure in one integration is isolated in health check."""
    registry = IntegrationRegistry()

    good_integration = DummyConfiguredIntegration(name="good_service")
    failing_integration = DummyFailingIntegration(name="failing_service")

    registry.register(good_integration)
    registry.register(failing_integration)

    statuses = await registry.check_health_all()

    assert statuses["good_service"].state == IntegrationState.CONNECTED
    assert statuses["good_service"].is_connected is True

    assert statuses["failing_service"].state == IntegrationState.ERROR
    assert statuses["failing_service"].is_connected is False
    assert "Upstream server unreachable" in str(statuses["failing_service"].error_message)


def test_six_planned_integrations_present_and_unconnected() -> None:
    """Requirements: Verify all 6 planned integrations are registered and do not claim connected."""
    from synqra.integrations import default_registry

    registered_names = default_registry.list_names()
    for planned in ALL_PLANNED_INTEGRATIONS:
        assert planned in registered_names, f"Integration '{planned}' must be registered"
        integration = default_registry.get(planned)
        assert integration is not None
        # Must not claim connected when credentials are not present
        assert integration.is_configured() is False


@pytest.mark.asyncio
async def test_extensible_registration_without_modifying_server() -> None:
    """Demonstrate how a new integration registers dynamically without server modifications."""
    custom_registry = IntegrationRegistry(planned=["custom_service"])
    custom_integration = DummyConfiguredIntegration(name="custom_service")
    custom_registry.register(custom_integration)

    # Server receives the custom registry without modifying server source code
    server = create_mcp_server(registry=custom_registry)

    # Tools should include custom integration's tool
    tools = await server.list_tools()
    tool_names = [t.name for t in tools]
    assert "custom_service_ping" in tool_names
    assert "health_check" in tool_names

    # Health check reflects the newly registered integration
    health_res = await server.call_tool("health_check", {})
    structured = health_res.structured_content
    assert "custom_service" in structured["available_integrations"]
    assert structured["integrations_status"]["custom_service"]["state"] == "connected"


def test_exception_hierarchy() -> None:
    """Verify custom exception inheritance and formatting."""
    err = IntegrationConfigurationError("Missing API key", integration_name="youtube")
    assert isinstance(err, SynqraIntegrationError)
    assert "[youtube] Missing API key" in str(err)
    assert err.integration_name == "youtube"
