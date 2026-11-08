"""Modern session-based Composio client for the YouTube integration.

Uses the official Composio Session API (`composio.create(user_id=..., toolkits=['youtube'])`)
to manage user-scoped tool sessions, verify account connections, discover tools,
and execute actions with full error isolation.
"""

import asyncio
import logging
from typing import Any

from composio import Composio
from composio.exceptions import ApiKeyError, ApiKeyNotProvidedError, ToolInputRequiredError

from synqra.core.config import Settings, get_settings
from synqra.integrations.exceptions import (
    IntegrationConfigurationError,
    IntegrationConnectionError,
    IntegrationExecutionError,
)

logger = logging.getLogger("synqra.integrations.youtube.client")

# Standard YouTube toolkit action slugs in Composio
YOUTUBE_SEARCH_ACTION = "YOUTUBE_SEARCH"
YOUTUBE_DETAILS_ACTION = "YOUTUBE_GET_VIDEO_DETAILS_BATCH"


class YouTubeClient:
    """Client coordinating YouTube operations through Composio's current Session API."""

    def __init__(
        self,
        api_key: str | None = None,
        user_id: str | None = None,
        settings: Settings | None = None,
        client: Composio | None = None,
        session: Any | None = None,
    ) -> None:
        """Initialize the YouTube client with stable user scoping.

        Args:
            api_key: Optional Composio API key. If omitted, loaded from settings.
            user_id: Stable user identifier. If omitted, loaded from settings.
            settings: Optional application settings.
            client: Optional pre-configured Composio instance (for testing/mocking).
            session: Optional pre-configured session instance (for testing/mocking).
        """
        self._settings = settings or get_settings()
        self.user_id = user_id or self._settings.composio_user_id

        resolved_key = api_key
        if resolved_key is None and self._settings.composio_api_key is not None:
            resolved_key = self._settings.composio_api_key.get_secret_value()

        self._api_key = resolved_key
        self._client = client
        self._session = session
        self._last_auth_url: str | None = None

    @property
    def is_configured(self) -> bool:
        """Return True if credentials or an active session/client are present."""
        return self._session is not None or self._client is not None or self._api_key is not None

    @property
    def authorization_url(self) -> str | None:
        """Return the most recent OAuth authorization URL for YouTube."""
        return self._last_auth_url

    def _get_composio_client(self) -> Composio:
        """Return the Composio client instance, lazily initializing if needed."""
        if self._client is not None:
            return self._client

        if not self._api_key:
            raise IntegrationConfigurationError(
                "Composio API key is not configured. Set SYNQRA_COMPOSIO_API_KEY in environment.",
                integration_name="youtube",
            )

        try:
            self._client = Composio(api_key=self._api_key)
            return self._client
        except (ApiKeyError, ApiKeyNotProvidedError) as exc:
            raise IntegrationConfigurationError(
                f"Invalid or missing Composio API key: {exc}",
                integration_name="youtube",
            ) from exc
        except Exception as exc:
            raise IntegrationConfigurationError(
                f"Failed to initialize Composio client: {exc}",
                integration_name="youtube",
            ) from exc

    def get_or_create_session(self) -> Any:
        """Create or return a user-scoped session restricted to the YouTube toolkit."""
        if self._session is not None:
            return self._session

        client = self._get_composio_client()
        try:
            self._session = client.create(
                user_id=self.user_id,
                toolkits=["youtube"],
            )
            return self._session
        except (ApiKeyError, ApiKeyNotProvidedError) as exc:
            raise IntegrationConfigurationError(
                f"Composio API key authentication failed: {exc}",
                integration_name="youtube",
            ) from exc
        except Exception as exc:
            raise IntegrationConnectionError(
                f"Failed to create Composio session for user '{self.user_id}': {exc}",
                integration_name="youtube",
            ) from exc

    def check_connection(self) -> bool:
        """Synchronously check whether an active YouTube connection exists for this user."""
        if not self.is_configured:
            return False

        try:
            session = self.get_or_create_session()
            details = session.toolkits(toolkits=["youtube"])
            items = getattr(details, "items", [])

            for item in items:
                slug = (getattr(item, "slug", "") or "").lower()
                connection = getattr(item, "connection", None)
                if (
                    slug == "youtube"
                    and connection is not None
                    and getattr(connection, "is_active", False) is True
                ):
                    return True

            return False
        except ApiKeyError:
            return False
        except Exception as exc:
            logger.warning("Error probing YouTube connection status: %s", exc)
            return False

    async def check_connection_async(self) -> bool:
        """Asynchronously verify active YouTube connection without blocking the event loop."""
        return await asyncio.to_thread(self.check_connection)

    def authorize(self, callback_url: str | None = None) -> str | None:
        """Initiate connection authorization for YouTube and obtain a redirect URL.

        Args:
            callback_url: Optional OAuth redirect callback URL.

        Returns:
            The authorization redirect URL for the user to connect their YouTube account.
        """
        session = self.get_or_create_session()
        try:
            req = session.authorize("youtube", callback_url=callback_url)
            self._last_auth_url = getattr(req, "redirect_url", None)
            return self._last_auth_url
        except Exception as exc:
            logger.warning("Failed to initiate YouTube authorization: %s", exc)
            raise IntegrationConnectionError(
                f"Failed to initiate YouTube authorization: {exc}",
                integration_name="youtube",
            ) from exc

    async def authorize_async(self, callback_url: str | None = None) -> str | None:
        """Asynchronously obtain YouTube authorization URL."""
        return await asyncio.to_thread(self.authorize, callback_url)

    def discover_tools(self) -> list[str]:
        """Discover available YouTube tool slugs dynamically through the session."""
        try:
            session = self.get_or_create_session()
            search_res = session.search(query="search youtube videos and get video details")
            slugs: list[str] = []
            for item in getattr(search_res, "items", []):
                slug = getattr(item, "slug", None)
                if slug:
                    slugs.append(slug)
            if slugs:
                return slugs
        except Exception as exc:
            logger.debug("Session tool discovery fallback: %s", exc)

        return [YOUTUBE_SEARCH_ACTION, YOUTUBE_DETAILS_ACTION]

    def execute_tool(self, tool_slug: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool action through the active user session.

        Args:
            tool_slug: Slugs such as 'YOUTUBE_SEARCH' or 'YOUTUBE_VIDEOS_LIST'.
            arguments: Input arguments for the tool.

        Returns:
            Dictionary containing the tool execution data.
        """
        session = self.get_or_create_session()

        try:
            response = session.execute(
                tool_slug=tool_slug,
                arguments=arguments,
            )
            result_type = getattr(response, "result_type", "completed")
            error_msg = getattr(response, "error", None)
            data = getattr(response, "data", None)

            if result_type in ("failed", "blocked") or error_msg:
                raise IntegrationExecutionError(
                    f"Composio tool execution failed ({result_type}): {error_msg}",
                    integration_name="youtube",
                )

            if isinstance(data, dict):
                return data
            return {"data": data}
        except ApiKeyError as exc:
            raise IntegrationConfigurationError(
                f"Composio API key invalid: {exc}",
                integration_name="youtube",
            ) from exc
        except ToolInputRequiredError as exc:
            raise IntegrationExecutionError(
                f"Tool requires user authorization or input: {exc}",
                integration_name="youtube",
            ) from exc
        except IntegrationExecutionError:
            raise
        except Exception as exc:
            raise IntegrationExecutionError(
                f"Error executing YouTube action '{tool_slug}': {exc}",
                integration_name="youtube",
            ) from exc

    async def execute_tool_async(self, tool_slug: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Asynchronously execute a YouTube action without blocking the event loop."""
        return await asyncio.to_thread(self.execute_tool, tool_slug, arguments)
