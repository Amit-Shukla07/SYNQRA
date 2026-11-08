"""Google Drive integration connector for SYNQRA.

Encapsulates Google Drive file discovery, metadata inspection, content downloads,
uploads, Google Docs operations, and folder management via Composio's session-based API.
"""

import logging

from fastmcp import FastMCP

from synqra.core.config import Settings, get_settings
from synqra.integrations.base import BaseIntegration
from synqra.integrations.google_drive.client import GoogleDriveClient
from synqra.integrations.google_drive.schemas import (
    CreateDriveFolderOutput,
    CreateGoogleDocumentOutput,
    DownloadDriveFileOutput,
    DriveFileDetails,
    SearchDriveFilesOutput,
    UpdateGoogleDocumentOutput,
    UploadDriveFileOutput,
)
from synqra.integrations.google_drive.tools import (
    create_drive_folder,
    create_google_document,
    download_drive_file,
    get_drive_file,
    search_drive_files,
    update_google_document,
    upload_drive_file,
)
from synqra.integrations.models import (
    IntegrationMetadata,
    IntegrationState,
    IntegrationStatus,
    utc_now,
)

logger = logging.getLogger("synqra.integrations.google_drive")


class GoogleDriveIntegration(BaseIntegration):
    """Production integration connector for Google Drive via Composio."""

    def __init__(
        self,
        client: GoogleDriveClient | None = None,
        settings: Settings | None = None,
        user_id: str | None = None,
    ) -> None:
        """Initialize the Google Drive integration instance.

        Args:
            client: Optional pre-configured GoogleDriveClient instance.
            settings: Optional application settings.
            user_id: Optional user identifier for Composio session scoping.
        """
        super().__init__(
            metadata=IntegrationMetadata(
                name="google_drive",
                display_name="Google Drive",
                description=(
                    "Search files and folders, retrieve file metadata, download contents, "
                    "upload files, create/update Google Docs, and manage folders."
                ),
                version="0.1.0",
                auth_required=True,
                auth_type="composio",
                documentation_url="https://developers.google.com/drive",
            )
        )
        self._settings = settings or get_settings()
        self.client = client or GoogleDriveClient(
            settings=self._settings,
            user_id=user_id or self._settings.composio_user_id,
        )

    def is_configured(self) -> bool:
        """Check whether Composio credentials are configured."""
        if getattr(self.client, "is_configured", False):
            return True
        return self._settings.composio_api_key is not None

    async def test_connection(self) -> bool:
        """Verify that a valid, active Google Drive connection exists through Composio."""
        if not self.is_configured():
            return False
        return await self.client.check_connection_async()

    async def health_check(self) -> IntegrationStatus:
        """Return operational health status, providing an authorization URL if unauthenticated."""
        if not self.is_configured():
            return IntegrationStatus(
                name=self.name,
                display_name=self.display_name,
                state=IntegrationState.UNCONFIGURED,
                is_configured=False,
                is_connected=False,
                registered_tools=[],
                error_message="Composio API key is not configured.",
                last_checked=utc_now(),
            )

        try:
            connected = await self.test_connection()
        except Exception as exc:
            logger.warning("Error probing connection for '%s': %s", self.name, exc)
            connected = False

        if connected:
            return IntegrationStatus(
                name=self.name,
                display_name=self.display_name,
                state=IntegrationState.CONNECTED,
                is_configured=True,
                is_connected=True,
                registered_tools=self._registered_tools,
                error_message=None,
                last_checked=utc_now(),
            )

        # Configured but not connected: obtain authorization redirect URL
        auth_url: str | None = None
        try:
            auth_url = await self.client.authorize_async()
        except Exception as exc:
            logger.debug("Failed to obtain Google Drive authorization URL: %s", exc)

        err_msg = (
            f"Google Drive account not connected. Authorize at: {auth_url}"
            if auth_url
            else "Google Drive account not connected. Authorize via Composio."
        )

        return IntegrationStatus(
            name=self.name,
            display_name=self.display_name,
            state=IntegrationState.CONFIGURED,
            is_configured=True,
            is_connected=False,
            registered_tools=[],
            error_message=err_msg,
            last_checked=utc_now(),
        )

    def _register_tools_impl(self, server: FastMCP) -> list[str]:
        """Register SYNQRA Google Drive MCP tools onto the FastMCP server."""

        @server.tool(
            name="search_drive_files",
            description=(
                "Search files and folders in Google Drive by name, MIME type, and query filters. "
                "Supports pagination and folder scoping."
            ),
        )
        async def search_drive_files_tool(
            query: str | None = None,
            name: str | None = None,
            mime_type: str | None = None,
            folder_id: str | None = None,
            page_size: int = 10,
        ) -> SearchDriveFilesOutput:
            """Search Google Drive files and folders."""
            return await search_drive_files(
                client=self.client,
                query=query,
                name=name,
                mime_type=mime_type,
                folder_id=folder_id,
                page_size=page_size,
            )

        @server.tool(
            name="get_drive_file",
            description=(
                "Retrieve detailed metadata for a Google Drive file or folder by ID, including "
                "name, MIME type, size, modification timestamp, web view URL, and parent folders."
            ),
        )
        async def get_drive_file_tool(
            file_id: str,
        ) -> DriveFileDetails:
            """Retrieve Google Drive file metadata."""
            return await get_drive_file(
                client=self.client,
                file_id=file_id,
            )

        @server.tool(
            name="download_drive_file",
            description=(
                "Retrieve file contents from Google Drive. Handles text files, binary files, "
                "and Google Workspace document exports (Docs, Sheets) safely."
            ),
        )
        async def download_drive_file_tool(
            file_id: str,
            mime_type: str | None = None,
        ) -> DownloadDriveFileOutput:
            """Download file contents from Google Drive."""
            return await download_drive_file(
                client=self.client,
                file_id=file_id,
                mime_type=mime_type,
            )

        @server.tool(
            name="upload_drive_file",
            description=(
                "Upload a text or structured file to Google Drive with configurable name, "
                "MIME type, and optional target parent folder."
            ),
        )
        async def upload_drive_file_tool(
            name: str,
            content: str,
            mime_type: str = "text/plain",
            parent_id: str | None = None,
        ) -> UploadDriveFileOutput:
            """Upload a file to Google Drive."""
            return await upload_drive_file(
                client=self.client,
                name=name,
                content=content,
                mime_type=mime_type,
                parent_id=parent_id,
            )

        @server.tool(
            name="create_google_document",
            description=(
                "Create a Google Docs document with supplied title and initial text content, "
                "optionally placed in a target folder."
            ),
        )
        async def create_google_document_tool(
            title: str,
            content: str,
            parent_id: str | None = None,
        ) -> CreateGoogleDocumentOutput:
            """Create a new Google Docs document."""
            return await create_google_document(
                client=self.client,
                title=title,
                content=content,
                parent_id=parent_id,
            )

        @server.tool(
            name="update_google_document",
            description=(
                "Update an existing Google Docs document or Drive file. Supports inserting text "
                "(at end or specific index) and replacing all occurrences of text."
            ),
        )
        async def update_google_document_tool(
            file_id: str,
            content: str,
            find_text: str | None = None,
            insertion_index: int | None = None,
            mime_type: str | None = None,
        ) -> UpdateGoogleDocumentOutput:
            """Update an existing Google Document or Drive file."""
            return await update_google_document(
                client=self.client,
                file_id=file_id,
                content=content,
                find_text=find_text,
                insertion_index=insertion_index,
                mime_type=mime_type,
            )

        @server.tool(
            name="create_drive_folder",
            description=(
                "Create a folder inside Google Drive, optionally under a specified parent folder."
            ),
        )
        async def create_drive_folder_tool(
            name: str,
            parent_id: str | None = None,
        ) -> CreateDriveFolderOutput:
            """Create a Google Drive folder."""
            return await create_drive_folder(
                client=self.client,
                name=name,
                parent_id=parent_id,
            )

        return [
            "search_drive_files",
            "get_drive_file",
            "download_drive_file",
            "upload_drive_file",
            "create_google_document",
            "update_google_document",
            "create_drive_folder",
        ]
