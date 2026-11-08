"""Standardized exception hierarchy for SYNQRA service integrations.

Ensures that errors arising from individual external integrations provide
clear context without compromising the stability of the overall MCP server.
"""


class SynqraIntegrationError(Exception):
    """Base exception for all integration-related errors in SYNQRA."""

    def __init__(self, message: str, integration_name: str | None = None) -> None:
        self.integration_name = integration_name
        prefix = f"[{integration_name}] " if integration_name else ""
        super().__init__(f"{prefix}{message}")


class IntegrationNotFoundError(SynqraIntegrationError):
    """Raised when an operation references an unregistered integration."""

    def __init__(self, integration_name: str) -> None:
        super().__init__(
            f"Integration '{integration_name}' is not registered.",
            integration_name=integration_name,
        )


class IntegrationConfigurationError(SynqraIntegrationError):
    """Raised when required credentials or configuration settings are missing or invalid."""

    def __init__(self, message: str, integration_name: str | None = None) -> None:
        super().__init__(message, integration_name=integration_name)


class IntegrationConnectionError(SynqraIntegrationError):
    """Raised when an integration fails to establish connection with an upstream service."""

    def __init__(self, message: str, integration_name: str | None = None) -> None:
        super().__init__(message, integration_name=integration_name)


class IntegrationExecutionError(SynqraIntegrationError):
    """Raised when an error occurs during execution of an integration tool."""

    def __init__(self, message: str, integration_name: str | None = None) -> None:
        super().__init__(message, integration_name=integration_name)


class IntegrationRegistrationError(SynqraIntegrationError):
    """Raised when an integration fails to register its tools with FastMCP."""

    def __init__(self, message: str, integration_name: str | None = None) -> None:
        super().__init__(message, integration_name=integration_name)
