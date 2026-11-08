"""Session-based Composio client for the Google Drive integration in SYNQRA.

Coordinates operations with the Google Drive toolkit using Composio's current Session API
(`composio.create(user_id=..., toolkits=['googledrive'])`), providing scoped execution,
authorization link generation, dynamic tool invocation, and complete error isolation.
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

logger = logging.getLogger("synqra.integrations.google_drive.client")

# Official Composio Google Drive & Docs action slugs verified from live runtime discovery
GOOGLEDRIVE_FIND_FILE_ACTION = "GOOGLEDRIVE_FIND_FILE"
GOOGLEDRIVE_GET_FILE_ACTION = "GOOGLEDRIVE_GET_FILE_METADATA"
GOOGLEDRIVE_DOWNLOAD_ACTION = "GOOGLEDRIVE_DOWNLOAD_FILE"
GOOGLEDRIVE_CREATE_TEXT_ACTION = "GOOGLEDRIVE_CREATE_FILE_FROM_TEXT"
GOOGLEDRIVE_EDIT_FILE_ACTION = "GOOGLEDRIVE_EDIT_FILE"
GOOGLEDRIVE_CREATE_FOLDER_ACTION = "GOOGLEDRIVE_CREATE_FOLDER"
GOOGLEDRIVE_GET_DOCUMENT_ACTION = "GOOGLEDRIVE_GET_DOCUMENT"

GOOGLEDOCS_INSERT_TEXT_ACTION = "GOOGLEDOCS_INSERT_TEXT_ACTION"
GOOGLEDOCS_REPLACE_ALL_TEXT_ACTION = "GOOGLEDOCS_REPLACE_ALL_TEXT"
GOOGLEDOCS_GET_PLAINTEXT_ACTION = "GOOGLEDOCS_GET_DOCUMENT_PLAINTEXT"


class GoogleDriveClient:
    """Client coordinating Google Drive operations through Composio's Session API."""

    def __init__(
        self,
        api_key: str | None = None,
        user_id: str | None = None,
        settings: Settings | None = None,
        client: Composio | None = None,
        session: Any | None = None,
    ) -> None:
        """Initialize the Google Drive client with stable user scoping.

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
        """Return True if Composio credentials or an active session/client are present."""
        return self._session is not None or self._client is not None or self._api_key is not None

    @property
    def authorization_url(self) -> str | None:
        """Return the most recent OAuth authorization URL for Google Drive."""
        return self._last_auth_url

    def _get_composio_client(self) -> Composio:
        """Return the Composio client instance, lazily initializing if needed."""
        if self._client is not None:
            return self._client

        if not self._api_key:
            raise IntegrationConfigurationError(
                "Composio API key is not configured. Set SYNQRA_COMPOSIO_API_KEY in environment.",
                integration_name="google_drive",
            )

        try:
            self._client = Composio(api_key=self._api_key)
            return self._client
        except (ApiKeyError, ApiKeyNotProvidedError) as exc:
            raise IntegrationConfigurationError(
                f"Invalid or missing Composio API key: {exc}",
                integration_name="google_drive",
            ) from exc
        except Exception as exc:
            raise IntegrationConfigurationError(
                f"Failed to initialize Composio client: {exc}",
                integration_name="google_drive",
            ) from exc

    def get_or_create_session(self) -> Any:
        """Create or return a user-scoped session restricted to the Google Drive toolkit."""
        if self._session is not None:
            return self._session

        client = self._get_composio_client()
        try:
            self._session = client.create(
                user_id=self.user_id,
                toolkits=["googledrive", "googledocs"],
            )
            return self._session
        except (ApiKeyError, ApiKeyNotProvidedError) as exc:
            raise IntegrationConfigurationError(
                f"Composio API key authentication failed: {exc}",
                integration_name="google_drive",
            ) from exc
        except Exception as exc:
            raise IntegrationConnectionError(
                f"Failed to create Composio session for user '{self.user_id}': {exc}",
                integration_name="google_drive",
            ) from exc

    def check_connection(self) -> bool:
        """Synchronously check whether an active Google Drive connection exists for this user."""
        if not self.is_configured:
            return False

        try:
            session = self.get_or_create_session()
            details = session.toolkits(toolkits=["googledrive"])
            items = getattr(details, "items", [])

            for item in items:
                slug = (getattr(item, "slug", "") or "").lower()
                connection = getattr(item, "connection", None)
                if (
                    slug in ("googledrive", "google_drive")
                    and connection is not None
                    and getattr(connection, "is_active", False) is True
                ):
                    return True

            return False
        except ApiKeyError:
            return False
        except Exception as exc:
            logger.warning("Error probing Google Drive connection status: %s", exc)
            return False

    async def check_connection_async(self) -> bool:
        """Asynchronously verify active Google Drive connection without blocking the event loop."""
        return await asyncio.to_thread(self.check_connection)

    def authorize(self, callback_url: str | None = None) -> str | None:
        """Initiate connection authorization for Google Drive and obtain a redirect URL.

        Args:
            callback_url: Optional OAuth redirect callback URL.

        Returns:
            The authorization redirect URL for the user to connect their Google Drive account.
        """
        session = self.get_or_create_session()
        try:
            req = session.authorize("googledrive", callback_url=callback_url)
            self._last_auth_url = getattr(req, "redirect_url", None)
            return self._last_auth_url
        except Exception as exc:
            logger.warning("Failed to initiate Google Drive authorization: %s", exc)
            raise IntegrationConnectionError(
                f"Failed to initiate Google Drive authorization: {exc}",
                integration_name="google_drive",
            ) from exc

    async def authorize_async(self, callback_url: str | None = None) -> str | None:
        """Asynchronously obtain Google Drive authorization URL."""
        return await asyncio.to_thread(self.authorize, callback_url)

    def execute_tool(self, tool_slug: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool action through the active user session.

        Args:
            tool_slug: Slugs such as 'GOOGLEDRIVE_FIND_FILE' or 'GOOGLEDRIVE_DOWNLOAD_FILE'.
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
                err_str = str(error_msg or "")
                if "rate limit" in err_str.lower() or "429" in err_str:
                    raise IntegrationExecutionError(
                        f"Google Drive API rate limit exceeded: {error_msg}",
                        integration_name="google_drive",
                    )
                if "not found" in err_str.lower() or "404" in err_str:
                    raise IntegrationExecutionError(
                        f"Requested Google Drive resource not found: {error_msg}",
                        integration_name="google_drive",
                    )
                if "permission" in err_str.lower() or "403" in err_str:
                    raise IntegrationExecutionError(
                        f"Insufficient permissions for Google Drive resource: {error_msg}",
                        integration_name="google_drive",
                    )
                if "no active connection" in err_str.lower() and "googledocs" in err_str.lower():
                    docs_url: str | None = None
                    try:
                        req = session.authorize("googledocs")
                        docs_url = getattr(req, "redirect_url", None)
                    except Exception:
                        pass
                    msg = (
                        f"Google Docs authorization required. Authorize at: {docs_url}"
                        if docs_url
                        else "Google Docs authorization required. Please authorize via Composio."
                    )
                    raise IntegrationExecutionError(msg, integration_name="google_drive")
                if "index" in err_str.lower() and (
                    "bound" in err_str.lower()
                    or "paragraph" in err_str.lower()
                    or "must be less" in err_str.lower()
                ):
                    raise IntegrationExecutionError(
                        f"Invalid text insertion index: {error_msg}",
                        integration_name="google_drive",
                    )
                raise IntegrationExecutionError(
                    f"Composio Google Drive tool execution failed ({result_type}): {error_msg}",
                    integration_name="google_drive",
                )

            if isinstance(data, dict):
                return data
            return {"data": data}
        except ApiKeyError as exc:
            raise IntegrationConfigurationError(
                f"Composio API key invalid: {exc}",
                integration_name="google_drive",
            ) from exc
        except ToolInputRequiredError as exc:
            raise IntegrationExecutionError(
                f"Tool requires user authorization or input: {exc}",
                integration_name="google_drive",
            ) from exc
        except IntegrationExecutionError:
            raise
        except Exception as exc:
            err_str = str(exc)
            if "no active connection" in err_str.lower() and "googledocs" in err_str.lower():
                raise IntegrationExecutionError(
                    "Google Docs authorization required to edit native Google Documents. "
                    "Please authorize the googledocs toolkit via Composio.",
                    integration_name="google_drive",
                ) from exc
            raise IntegrationExecutionError(
                f"Error executing Google Drive action '{tool_slug}': {exc}",
                integration_name="google_drive",
            ) from exc

    async def execute_tool_async(self, tool_slug: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Asynchronously execute a Google Drive action without blocking the event loop."""
        return await asyncio.to_thread(self.execute_tool, tool_slug, arguments)
