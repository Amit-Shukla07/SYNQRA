"""SYNQRA Integration Architecture.

Provides the common base abstraction, registry, status models, and exceptions
for all external service integrations (YouTube, Google Drive, GitHub, Google Calendar,
Web Search, Gmail).
"""

from synqra.integrations.base import BaseIntegration
from synqra.integrations.exceptions import (
    IntegrationConfigurationError,
    IntegrationConnectionError,
    IntegrationExecutionError,
    IntegrationNotFoundError,
    IntegrationRegistrationError,
    SynqraIntegrationError,
)
from synqra.integrations.github import GitHubIntegration
from synqra.integrations.gmail import GmailIntegration
from synqra.integrations.google_calendar import GoogleCalendarIntegration
from synqra.integrations.google_drive import GoogleDriveIntegration
from synqra.integrations.models import (
    IntegrationMetadata,
    IntegrationState,
    IntegrationStatus,
)
from synqra.integrations.registry import (
    ALL_PLANNED_INTEGRATIONS,
    IntegrationRegistry,
    default_registry,
)
from synqra.integrations.web_search import WebSearchIntegration
from synqra.integrations.youtube import YouTubeIntegration

# Register the 6 planned service integrations with their metadata
default_registry.register(YouTubeIntegration())
default_registry.register(GoogleDriveIntegration())
default_registry.register(GitHubIntegration())
default_registry.register(GoogleCalendarIntegration())
default_registry.register(WebSearchIntegration())
default_registry.register(GmailIntegration())

__all__ = [
    "ALL_PLANNED_INTEGRATIONS",
    "BaseIntegration",
    "GitHubIntegration",
    "GmailIntegration",
    "GoogleCalendarIntegration",
    "GoogleDriveIntegration",
    "IntegrationConfigurationError",
    "IntegrationConnectionError",
    "IntegrationExecutionError",
    "IntegrationMetadata",
    "IntegrationNotFoundError",
    "IntegrationRegistrationError",
    "IntegrationRegistry",
    "IntegrationState",
    "IntegrationStatus",
    "SynqraIntegrationError",
    "WebSearchIntegration",
    "YouTubeIntegration",
    "default_registry",
]
