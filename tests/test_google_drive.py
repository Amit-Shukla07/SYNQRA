"""Unit and integration tests for SYNQRA Google Drive integration.

Verifies GoogleDriveClient, schema models, tools, and integration health checks using mocked
Composio sessions without requiring live external network credentials.
"""

from unittest.mock import MagicMock

import pytest
from fastmcp import FastMCP

from synqra.core.config import Settings
from synqra.integrations.exceptions import (
    IntegrationConfigurationError,
    IntegrationExecutionError,
)
from synqra.integrations.google_drive.client import (
    GOOGLEDOCS_GET_PLAINTEXT_ACTION,
    GOOGLEDOCS_INSERT_TEXT_ACTION,
    GOOGLEDOCS_REPLACE_ALL_TEXT_ACTION,
    GOOGLEDRIVE_CREATE_FOLDER_ACTION,
    GOOGLEDRIVE_CREATE_TEXT_ACTION,
    GOOGLEDRIVE_DOWNLOAD_ACTION,
    GOOGLEDRIVE_EDIT_FILE_ACTION,
    GOOGLEDRIVE_FIND_FILE_ACTION,
    GOOGLEDRIVE_GET_DOCUMENT_ACTION,
    GOOGLEDRIVE_GET_FILE_ACTION,
    GoogleDriveClient,
)
from synqra.integrations.google_drive.integration import GoogleDriveIntegration
from synqra.integrations.google_drive.tools import (
    create_drive_folder,
    create_google_document,
    download_drive_file,
    get_drive_file,
    read_google_document,
    search_drive_files,
    update_google_document,
    upload_drive_file,
)
from synqra.integrations.models import IntegrationState

# ==============================================================================
# Client & Configuration Tests
# ==============================================================================


def test_google_drive_client_unconfigured_without_api_key() -> None:
    """Verify GoogleDriveClient reports unconfigured when no Composio key is provided."""
    empty_settings = Settings()
    client = GoogleDriveClient(settings=empty_settings)

    assert client.is_configured is False
    with pytest.raises(IntegrationConfigurationError) as exc_info:
        client._get_composio_client()
    assert "Composio API key is not configured" in str(exc_info.value)


def test_google_drive_client_configured_with_api_key() -> None:
    """Verify GoogleDriveClient reports configured when Composio key is present."""
    client = GoogleDriveClient(api_key="ak_mock_key_12345")
    assert client.is_configured is True
    assert client.user_id == "synqra_default_user"


def test_google_drive_client_check_connection_active() -> None:
    """Verify check_connection returns True when toolkit connection is active."""
    mock_session = MagicMock()
    mock_item = MagicMock()
    mock_item.slug = "googledrive"
    mock_item.connection = MagicMock(is_active=True)
    mock_session.toolkits.return_value = MagicMock(items=[mock_item])

    client = GoogleDriveClient(api_key="ak_mock_key", session=mock_session)
    assert client.check_connection() is True


def test_google_drive_client_check_connection_inactive() -> None:
    """Verify check_connection returns False when toolkit connection is inactive."""
    mock_session = MagicMock()
    mock_item = MagicMock()
    mock_item.slug = "googledrive"
    mock_item.connection = MagicMock(is_active=False)
    mock_session.toolkits.return_value = MagicMock(items=[mock_item])

    client = GoogleDriveClient(api_key="ak_mock_key", session=mock_session)
    assert client.check_connection() is False


def test_google_drive_client_check_connection_error_returns_false() -> None:
    """Verify check_connection returns False when an unexpected exception occurs."""
    mock_session = MagicMock()
    mock_session.toolkits.side_effect = RuntimeError("Network timeout")

    client = GoogleDriveClient(api_key="ak_mock_key", session=mock_session)
    assert client.check_connection() is False


def test_google_drive_client_authorize_generates_redirect_url() -> None:
    """Verify authorize returns redirect URL for OAuth connection."""
    mock_session = MagicMock()
    mock_session.authorize.return_value = MagicMock(
        redirect_url="https://connect.composio.dev/link/gdrive_oauth_123"
    )

    client = GoogleDriveClient(api_key="ak_mock_key", session=mock_session)
    url = client.authorize(callback_url="http://localhost:8000/callback")

    assert url == "https://connect.composio.dev/link/gdrive_oauth_123"
    assert client.authorization_url == url


def test_google_drive_client_execute_tool_success() -> None:
    """Verify execute_tool returns data dictionary on completed result."""
    mock_session = MagicMock()
    mock_session.execute.return_value = MagicMock(
        result_type="completed",
        error=None,
        data={"files": [{"id": "f_1", "name": "Document.txt"}]},
    )

    client = GoogleDriveClient(api_key="ak_mock_key", session=mock_session)
    data = client.execute_tool(GOOGLEDRIVE_FIND_FILE_ACTION, {"q": "name = 'test'"})

    assert data == {"files": [{"id": "f_1", "name": "Document.txt"}]}


def test_google_drive_client_execute_tool_failed_raises() -> None:
    """Verify execute_tool identifies failures and raises IntegrationExecutionError."""
    mock_session = MagicMock()
    mock_session.execute.return_value = MagicMock(
        result_type="failed",
        error="File not found or permission denied",
        data=None,
    )

    client = GoogleDriveClient(api_key="ak_mock_key", session=mock_session)
    with pytest.raises(IntegrationExecutionError) as exc_info:
        client.execute_tool(GOOGLEDRIVE_GET_FILE_ACTION, {"fileId": "missing"})
    assert "permission denied" in str(exc_info.value).lower()


# ==============================================================================
# Tool 1: search_drive_files Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_search_drive_files_success() -> None:
    """Verify search_drive_files query building and normalization."""
    mock_client = MagicMock(spec=GoogleDriveClient)
    mock_client.execute_tool_async.return_value = {
        "files": [
            {
                "id": "file_101",
                "name": "Machine_Learning_Notes.pdf",
                "mimeType": "application/pdf",
                "size": "204800",
                "modifiedTime": "2026-03-01T12:00:00Z",
                "createdTime": "2026-03-01T10:00:00Z",
                "webViewLink": "https://drive.google.com/file/d/file_101/view",
                "parents": ["folder_001"],
            }
        ]
    }

    result = await search_drive_files(
        client=mock_client,
        query="Machine Learning",
        mime_type="application/pdf",
        folder_id="folder_001",
        page_size=10,
    )

    assert result.total_count == 1
    assert len(result.files) == 1
    file_item = result.files[0]
    assert file_item.id == "file_101"
    assert file_item.name == "Machine_Learning_Notes.pdf"
    assert file_item.mime_type == "application/pdf"
    assert file_item.size == 204800
    assert file_item.web_view_url == "https://drive.google.com/file/d/file_101/view"

    mock_client.execute_tool_async.assert_called_once()
    call_kwargs = mock_client.execute_tool_async.call_args.kwargs
    assert call_kwargs["tool_slug"] == GOOGLEDRIVE_FIND_FILE_ACTION
    params = call_kwargs["arguments"]
    assert "name contains 'Machine Learning'" in params["q"]
    assert "mimeType = 'application/pdf'" in params["q"]
    assert params["folder_id"] == "folder_001"


@pytest.mark.asyncio
async def test_search_drive_files_empty_results() -> None:
    """Verify search_drive_files handles empty search response."""
    mock_client = MagicMock(spec=GoogleDriveClient)
    mock_client.execute_tool_async.return_value = {"files": []}

    result = await search_drive_files(client=mock_client, query="Nonexistent")
    assert result.total_count == 0
    assert result.files == []


# ==============================================================================
# Tool 2: get_drive_file Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_get_drive_file_success() -> None:
    """Verify get_drive_file metadata retrieval and schema normalization."""
    mock_client = MagicMock(spec=GoogleDriveClient)
    mock_client.execute_tool_async.return_value = {
        "id": "file_202",
        "name": "Project_Roadmap.docx",
        "mimeType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "size": "51200",
        "modifiedTime": "2026-03-02T15:30:00Z",
        "createdTime": "2026-03-01T08:00:00Z",
        "webViewLink": "https://drive.google.com/file/d/file_202/view",
        "parents": ["root"],
    }

    details = await get_drive_file(client=mock_client, file_id="file_202")
    assert details.id == "file_202"
    assert details.name == "Project_Roadmap.docx"
    assert details.size == 51200
    assert details.web_view_url == "https://drive.google.com/file/d/file_202/view"

    mock_client.execute_tool_async.assert_called_once_with(
        tool_slug=GOOGLEDRIVE_GET_FILE_ACTION,
        arguments={
            "fileId": "file_202",
            "fields": "id,name,mimeType,size,modifiedTime,createdTime,webViewLink,parents,trashed",
        },
    )


@pytest.mark.asyncio
async def test_get_drive_file_empty_id_raises() -> None:
    """Verify get_drive_file raises IntegrationExecutionError if file_id is blank."""
    mock_client = MagicMock(spec=GoogleDriveClient)
    with pytest.raises(IntegrationExecutionError, match="File ID must not be empty"):
        await get_drive_file(client=mock_client, file_id="")


# ==============================================================================
# Tool 3: download_drive_file Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_download_drive_file_text_success() -> None:
    """Verify download_drive_file returns decoded text for text files."""
    mock_client = MagicMock(spec=GoogleDriveClient)
    mock_client.execute_tool_async.return_value = {
        "content": "Hello SYNQRA Google Drive!",
        "mimeType": "text/plain",
        "name": "greeting.txt",
    }

    result = await download_drive_file(client=mock_client, file_id="f_text_1")
    assert result.file_id == "f_text_1"
    assert result.content == "Hello SYNQRA Google Drive!"
    assert result.is_binary is False
    assert result.mime_type == "text/plain"


@pytest.mark.asyncio
async def test_download_drive_file_binary_success() -> None:
    """Verify download_drive_file returns binary flag for non-decodable byte content."""
    mock_client = MagicMock(spec=GoogleDriveClient)
    mock_client.execute_tool_async.return_value = {
        "content": b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01",
        "mimeType": "image/png",
        "name": "pixel.png",
    }

    result = await download_drive_file(client=mock_client, file_id="f_bin_1")
    assert result.file_id == "f_bin_1"
    assert result.is_binary is True
    assert result.content is None
    assert result.encoding == "binary"
    assert result.mime_type == "image/png"


@pytest.mark.asyncio
async def test_download_drive_file_workspace_export() -> None:
    """Verify download_drive_file passes requested export mime_type to action."""
    mock_client = MagicMock(spec=GoogleDriveClient)
    mock_client.execute_tool_async.return_value = {
        "content": "Exported doc text content",
        "mimeType": "text/plain",
    }

    result = await download_drive_file(
        client=mock_client,
        file_id="doc_id_999",
        mime_type="text/plain",
    )
    assert result.content == "Exported doc text content"
    mock_client.execute_tool_async.assert_called_once_with(
        tool_slug=GOOGLEDRIVE_DOWNLOAD_ACTION,
        arguments={"fileId": "doc_id_999", "mime_type": "text/plain"},
    )


# ==============================================================================
# Tool 4: upload_drive_file Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_upload_drive_file_success() -> None:
    """Verify upload_drive_file creates file with given text content."""
    mock_client = MagicMock(spec=GoogleDriveClient)
    mock_client.execute_tool_async.return_value = {
        "id": "new_file_id_123",
        "name": "notes.md",
        "mimeType": "text/markdown",
        "webViewLink": "https://drive.google.com/file/d/new_file_id_123/view",
    }

    result = await upload_drive_file(
        client=mock_client,
        name="notes.md",
        content="# Hello World\nSynqra integration",
        mime_type="text/markdown",
        parent_id="folder_99",
    )

    assert result.id == "new_file_id_123"
    assert result.name == "notes.md"
    assert result.web_view_url == "https://drive.google.com/file/d/new_file_id_123/view"

    mock_client.execute_tool_async.assert_called_once_with(
        tool_slug=GOOGLEDRIVE_CREATE_TEXT_ACTION,
        arguments={
            "file_name": "notes.md",
            "text_content": "# Hello World\nSynqra integration",
            "mime_type": "text/markdown",
            "parent_id": "folder_99",
        },
    )


@pytest.mark.asyncio
async def test_upload_drive_file_empty_name_raises() -> None:
    """Verify upload_drive_file raises IntegrationExecutionError if name is blank."""
    mock_client = MagicMock(spec=GoogleDriveClient)
    with pytest.raises(IntegrationExecutionError, match="File name must not be empty"):
        await upload_drive_file(client=mock_client, name="", content="abc")


# ==============================================================================
# Tool 5: create_google_document Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_create_google_document_success() -> None:
    """Verify create_google_document specifies Docs mimeType and returns details."""
    mock_client = MagicMock(spec=GoogleDriveClient)
    mock_client.execute_tool_async.return_value = {
        "id": "doc_id_456",
        "name": "Design Document",
        "webViewLink": "https://docs.google.com/document/d/doc_id_456/edit",
    }

    result = await create_google_document(
        client=mock_client,
        title="Design Document",
        content="System Architecture Overview",
    )

    assert result.document_id == "doc_id_456"
    assert result.title == "Design Document"
    assert result.web_view_url == "https://docs.google.com/document/d/doc_id_456/edit"

    mock_client.execute_tool_async.assert_called_once_with(
        tool_slug=GOOGLEDRIVE_CREATE_TEXT_ACTION,
        arguments={
            "file_name": "Design Document",
            "text_content": "System Architecture Overview",
            "mime_type": "application/vnd.google-apps.document",
        },
    )


@pytest.mark.asyncio
async def test_create_google_document_empty_title_raises() -> None:
    """Verify create_google_document raises IntegrationExecutionError if title is blank."""
    mock_client = MagicMock(spec=GoogleDriveClient)
    with pytest.raises(IntegrationExecutionError, match="Document title must not be empty"):
        await create_google_document(client=mock_client, title="")


# ==============================================================================
# Tool 6: update_google_document Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_update_google_document_drive_file_preservation() -> None:
    """Verify update_google_document preserves Google Drive text file editing."""
    mock_client = MagicMock(spec=GoogleDriveClient)
    mock_client.execute_tool_async.return_value = {
        "id": "file_txt_123",
        "name": "notes.txt",
        "modifiedTime": "2026-03-02T16:00:00Z",
    }

    result = await update_google_document(
        client=mock_client,
        file_id="file_txt_123",
        content="Updated text file content",
        mime_type="text/plain",
    )

    assert result.file_id == "file_txt_123"
    assert result.status == "updated"
    assert result.target_type == "drive_file"

    mock_client.execute_tool_async.assert_called_once_with(
        tool_slug=GOOGLEDRIVE_EDIT_FILE_ACTION,
        arguments={
            "file_id": "file_txt_123",
            "content": "Updated text file content",
            "mime_type": "text/plain",
        },
    )


@pytest.mark.asyncio
async def test_update_google_document_native_doc_insert_append() -> None:
    """Verify update_google_document routes Google Docs to GOOGLEDOCS_INSERT_TEXT_ACTION."""
    mock_client = MagicMock(spec=GoogleDriveClient)

    async def side_effect(tool_slug: str, arguments: dict, **_kwargs):
        _ = arguments
        if tool_slug == GOOGLEDOCS_INSERT_TEXT_ACTION:
            return {"documentId": "doc_native_001", "status": "completed"}
        if tool_slug == GOOGLEDOCS_GET_PLAINTEXT_ACTION:
            return {"text": "Original text.\nAppended paragraph."}
        return {}

    mock_client.execute_tool_async.side_effect = side_effect

    result = await update_google_document(
        client=mock_client,
        file_id="doc_native_001",
        content="\nAppended paragraph.",
        mime_type="application/vnd.google-apps.document",
    )

    assert result.file_id == "doc_native_001"
    assert result.status == "inserted"
    assert result.target_type == "google_doc"
    assert result.updated_content_preview == "Original text.\nAppended paragraph."

    calls = mock_client.execute_tool_async.call_args_list
    assert calls[0].kwargs["tool_slug"] == GOOGLEDOCS_INSERT_TEXT_ACTION
    assert calls[0].kwargs["arguments"]["document_id"] == "doc_native_001"
    assert calls[0].kwargs["arguments"]["text_to_insert"] == "\nAppended paragraph."
    assert calls[0].kwargs["arguments"]["append_to_end"] is True


@pytest.mark.asyncio
async def test_update_google_document_native_doc_insert_at_index() -> None:
    """Verify update_google_document inserts text at specific index when requested."""
    mock_client = MagicMock(spec=GoogleDriveClient)

    async def side_effect(tool_slug: str, arguments: dict, **_kwargs):
        _ = arguments
        if tool_slug == GOOGLEDOCS_INSERT_TEXT_ACTION:
            return {"documentId": "doc_native_002", "status": "completed"}
        if tool_slug == GOOGLEDOCS_GET_PLAINTEXT_ACTION:
            return {"text": "Prefix inserted. Body text."}
        return {}

    mock_client.execute_tool_async.side_effect = side_effect

    result = await update_google_document(
        client=mock_client,
        file_id="doc_native_002",
        content="Prefix inserted. ",
        insertion_index=1,
        mime_type="application/vnd.google-apps.document",
    )

    assert result.file_id == "doc_native_002"
    assert result.status == "inserted"
    assert result.target_type == "google_doc"

    calls = mock_client.execute_tool_async.call_args_list
    assert calls[0].kwargs["tool_slug"] == GOOGLEDOCS_INSERT_TEXT_ACTION
    assert calls[0].kwargs["arguments"]["document_id"] == "doc_native_002"
    assert calls[0].kwargs["arguments"]["text_to_insert"] == "Prefix inserted. "
    assert calls[0].kwargs["arguments"]["insertion_index"] == 1


@pytest.mark.asyncio
async def test_update_google_document_native_doc_replace_text() -> None:
    """Verify update_google_document replaces text via GOOGLEDOCS_REPLACE_ALL_TEXT."""
    mock_client = MagicMock(spec=GoogleDriveClient)

    async def side_effect(tool_slug: str, arguments: dict, **_kwargs):
        _ = arguments
        if tool_slug == GOOGLEDOCS_REPLACE_ALL_TEXT_ACTION:
            return {"documentId": "doc_native_003", "occurrencesChanged": 2}
        if tool_slug == GOOGLEDOCS_GET_PLAINTEXT_ACTION:
            return {"text": "Updated content after replacement."}
        return {}

    mock_client.execute_tool_async.side_effect = side_effect

    result = await update_google_document(
        client=mock_client,
        file_id="doc_native_003",
        content="New text",
        find_text="Old text",
        mime_type="application/vnd.google-apps.document",
    )

    assert result.file_id == "doc_native_003"
    assert result.status == "replaced"
    assert result.target_type == "google_doc"

    calls = mock_client.execute_tool_async.call_args_list
    assert calls[0].kwargs["tool_slug"] == GOOGLEDOCS_REPLACE_ALL_TEXT_ACTION
    assert calls[0].kwargs["arguments"]["document_id"] == "doc_native_003"
    assert calls[0].kwargs["arguments"]["find_text"] == "Old text"
    assert calls[0].kwargs["arguments"]["replace_text"] == "New text"


@pytest.mark.asyncio
async def test_update_google_document_auto_detects_mime_type() -> None:
    """Verify update_google_document queries metadata when mime_type is omitted."""
    mock_client = MagicMock(spec=GoogleDriveClient)

    async def side_effect(tool_slug: str, arguments: dict, **_kwargs):
        _ = arguments
        if tool_slug == GOOGLEDRIVE_GET_FILE_ACTION:
            return {
                "id": "doc_auto_004",
                "name": "AutoDoc",
                "mimeType": "application/vnd.google-apps.document",
            }
        if tool_slug == GOOGLEDOCS_INSERT_TEXT_ACTION:
            return {"documentId": "doc_auto_004"}
        if tool_slug == GOOGLEDOCS_GET_PLAINTEXT_ACTION:
            return {"text": "Auto-routed content."}
        return {}

    mock_client.execute_tool_async.side_effect = side_effect

    result = await update_google_document(
        client=mock_client,
        file_id="doc_auto_004",
        content="Auto-routed content.",
        mime_type=None,
    )

    assert result.target_type == "google_doc"
    assert result.status == "inserted"


@pytest.mark.asyncio
async def test_read_google_document_fallback() -> None:
    """Verify read_google_document falls back to GOOGLEDRIVE_GET_DOCUMENT on plaintext failure."""
    mock_client = MagicMock(spec=GoogleDriveClient)

    async def side_effect(tool_slug: str, arguments: dict, **_kwargs):
        _ = arguments
        if tool_slug == GOOGLEDOCS_GET_PLAINTEXT_ACTION:
            raise RuntimeError("Toolkit not connected")
        if tool_slug == GOOGLEDRIVE_GET_DOCUMENT_ACTION:
            return {
                "documentId": "doc_fallback_005",
                "title": "FallbackDoc",
                "body": {
                    "content": [
                        {
                            "paragraph": {
                                "elements": [
                                    {"textRun": {"content": "Fallback document content.\n"}}
                                ]
                            }
                        }
                    ]
                },
            }
        return {}

    mock_client.execute_tool_async.side_effect = side_effect

    text = await read_google_document(mock_client, "doc_fallback_005")
    assert text == "Fallback document content."


@pytest.mark.asyncio
async def test_update_google_document_empty_id_raises() -> None:
    """Verify update_google_document raises IntegrationExecutionError if file_id is blank."""
    mock_client = MagicMock(spec=GoogleDriveClient)
    with pytest.raises(IntegrationExecutionError, match="File ID must not be empty"):
        await update_google_document(client=mock_client, file_id="", content="x")


# ==============================================================================
# Tool 7: create_drive_folder Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_create_drive_folder_success() -> None:
    """Verify create_drive_folder creates folder with parent scoping."""
    mock_client = MagicMock(spec=GoogleDriveClient)
    mock_client.execute_tool_async.return_value = {
        "id": "folder_789",
        "name": "SYNQRA Workspace",
        "webViewLink": "https://drive.google.com/drive/folders/folder_789",
        "parent_id": "root_folder",
    }

    result = await create_drive_folder(
        client=mock_client,
        name="SYNQRA Workspace",
        parent_id="root_folder",
    )

    assert result.folder_id == "folder_789"
    assert result.name == "SYNQRA Workspace"
    assert result.parent_id == "root_folder"

    mock_client.execute_tool_async.assert_called_once_with(
        tool_slug=GOOGLEDRIVE_CREATE_FOLDER_ACTION,
        arguments={"name": "SYNQRA Workspace", "parent_id": "root_folder"},
    )


@pytest.mark.asyncio
async def test_create_drive_folder_empty_name_raises() -> None:
    """Verify create_drive_folder raises IntegrationExecutionError if name is blank."""
    mock_client = MagicMock(spec=GoogleDriveClient)
    with pytest.raises(IntegrationExecutionError, match="Folder name must not be empty"):
        await create_drive_folder(client=mock_client, name="")


# ==============================================================================
# Integration Lifecycle & Health Check Tests
# ==============================================================================


def test_google_drive_integration_metadata() -> None:
    """Verify GoogleDriveIntegration presents correct metadata."""
    integration = GoogleDriveIntegration()
    assert integration.name == "google_drive"
    assert integration.display_name == "Google Drive"
    assert integration.metadata.auth_type == "composio"
    assert integration.metadata.auth_required is True


@pytest.mark.asyncio
async def test_google_drive_integration_health_unconfigured() -> None:
    """Verify health check returns UNCONFIGURED when no Composio key is provided."""
    empty_settings = Settings()
    integration = GoogleDriveIntegration(settings=empty_settings)
    status = await integration.health_check()

    assert status.state == IntegrationState.UNCONFIGURED
    assert status.is_configured is False
    assert status.is_connected is False
    assert status.error_message is not None
    assert "not configured" in status.error_message


@pytest.mark.asyncio
async def test_google_drive_integration_health_configured_not_connected() -> None:
    """Verify health check returns CONFIGURED and provides auth URL when not connected."""
    mock_client = MagicMock(spec=GoogleDriveClient)
    mock_client.is_configured = True
    mock_client.check_connection_async.return_value = False
    mock_client.authorize_async.return_value = "https://connect.composio.dev/link/gdrive_oauth"

    integration = GoogleDriveIntegration(client=mock_client)
    status = await integration.health_check()

    assert status.state == IntegrationState.CONFIGURED
    assert status.is_configured is True
    assert status.is_connected is False
    assert "https://connect.composio.dev/link/gdrive_oauth" in str(status.error_message)


@pytest.mark.asyncio
async def test_google_drive_integration_health_connected() -> None:
    """Verify health check returns CONNECTED when OAuth account is active."""
    mock_client = MagicMock(spec=GoogleDriveClient)
    mock_client.is_configured = True
    mock_client.check_connection_async.return_value = True

    integration = GoogleDriveIntegration(client=mock_client)
    status = await integration.health_check()

    assert status.state == IntegrationState.CONNECTED
    assert status.is_configured is True
    assert status.is_connected is True
    assert status.error_message is None


def test_google_drive_integration_registers_7_tools() -> None:
    """Verify GoogleDriveIntegration mounts exactly 7 curated tools onto FastMCP."""
    server = FastMCP("test_synqra")
    mock_client = MagicMock(spec=GoogleDriveClient)
    mock_client.is_configured = True

    integration = GoogleDriveIntegration(client=mock_client)
    registered_tools = integration.register_tools(server)

    expected_tools = [
        "search_drive_files",
        "get_drive_file",
        "download_drive_file",
        "upload_drive_file",
        "create_google_document",
        "update_google_document",
        "create_drive_folder",
    ]

    assert len(registered_tools) == 7
    for tool_name in expected_tools:
        assert tool_name in registered_tools
