"""Tool implementations for the Google Drive integration in SYNQRA.

Provides standardized, well-structured tools for file search, metadata retrieval,
content downloads, uploads, Google Docs creation and updating, and folder creation via Composio.
"""

import logging
from typing import Any

from synqra.integrations.exceptions import IntegrationExecutionError
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
from synqra.integrations.google_drive.schemas import (
    GOOGLE_DOCS_MIME_TYPE,
    GOOGLE_FOLDER_MIME_TYPE,
    CreateDriveFolderOutput,
    CreateGoogleDocumentOutput,
    DownloadDriveFileOutput,
    DriveFileDetails,
    DriveFileSummary,
    SearchDriveFilesOutput,
    UpdateGoogleDocumentOutput,
    UploadDriveFileOutput,
)

logger = logging.getLogger("synqra.integrations.google_drive.tools")


def _extract_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract item records from various Composio Google Drive response envelopes."""
    if not isinstance(payload, dict):
        return []

    if "files" in payload and isinstance(payload["files"], list):
        return payload["files"]

    if "items" in payload and isinstance(payload["items"], list):
        return payload["items"]

    data = payload.get("data")
    if isinstance(data, dict):
        if "files" in data and isinstance(data["files"], list):
            return data["files"]
        if "items" in data and isinstance(data["items"], list):
            return data["items"]

    if isinstance(data, list):
        return data

    results = payload.get("results")
    if isinstance(results, list):
        return results

    return []


def _extract_dict(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract a single resource dictionary from various Composio response structures."""
    if not isinstance(payload, dict):
        return {}
    data = payload.get("data")
    if isinstance(data, dict):
        return data
    return payload


async def search_drive_files(
    client: GoogleDriveClient,
    query: str | None = None,
    name: str | None = None,
    mime_type: str | None = None,
    folder_id: str | None = None,
    page_size: int = 10,
) -> SearchDriveFilesOutput:
    """Search files and folders in Google Drive by name, MIME type, and query filters.

    Args:
        client: Configured GoogleDriveClient instance.
        query: Optional free-text or Drive API query filter.
        name: Optional filter for file name.
        mime_type: Optional MIME type filter.
        folder_id: Optional folder ID to restrict the search within.
        page_size: Number of files to return (1-100).

    Returns:
        SearchDriveFilesOutput with normalized file summaries.
    """
    limit = max(1, min(page_size, 100))
    query_parts: list[str] = ["trashed = false"]

    if name:
        clean_name = name.strip().replace("'", "\\'")
        query_parts.append(f"name contains '{clean_name}'")

    if mime_type:
        clean_mime = mime_type.strip().replace("'", "\\'")
        query_parts.append(f"mimeType = '{clean_mime}'")

    if query and query.strip():
        clean_q = query.strip()
        # If query already looks like a Drive query condition, include it directly
        if "=" in clean_q or "contains" in clean_q:
            query_parts.append(clean_q)
        else:
            escaped_q = clean_q.replace("'", "\\'")
            query_parts.append(f"(name contains '{escaped_q}' or fullText contains '{escaped_q}')")

    combined_q = " and ".join(query_parts)

    arguments: dict[str, Any] = {
        "q": combined_q,
        "pageSize": limit,
    }
    if folder_id:
        arguments["folder_id"] = folder_id.strip()

    raw_response = await client.execute_tool_async(
        tool_slug=GOOGLEDRIVE_FIND_FILE_ACTION,
        arguments=arguments,
    )

    items = _extract_items(raw_response)
    file_summaries: list[DriveFileSummary] = []

    for item in items:
        file_id = item.get("id") or item.get("file_id") or ""
        if not file_id:
            continue

        raw_size = item.get("size")
        size_int = int(raw_size) if raw_size and str(raw_size).isdigit() else None
        item_mime = item.get("mimeType") or item.get("mime_type") or "application/octet-stream"
        is_folder = item_mime == GOOGLE_FOLDER_MIME_TYPE

        file_summaries.append(
            DriveFileSummary(
                id=file_id,
                name=item.get("name") or item.get("title") or "Untitled",
                mime_type=item_mime,
                size=size_int,
                modified_time=item.get("modifiedTime") or item.get("modified_time"),
                web_view_url=item.get("webViewLink") or item.get("web_view_url"),
                is_folder=is_folder,
            )
        )

    return SearchDriveFilesOutput(
        total_count=len(file_summaries),
        files=file_summaries,
    )


async def get_drive_file(
    client: GoogleDriveClient,
    file_id: str,
) -> DriveFileDetails:
    """Retrieve detailed metadata for an individual Google Drive file or folder.

    Args:
        client: Configured GoogleDriveClient instance.
        file_id: The unique identifier of the file or folder.

    Returns:
        DriveFileDetails model with complete metadata.

    Raises:
        IntegrationExecutionError: If file ID is empty or file is not found.
    """
    clean_id = file_id.strip()
    if not clean_id:
        raise IntegrationExecutionError(
            "File ID must not be empty.",
            integration_name="google_drive",
        )

    arguments = {
        "fileId": clean_id,
        "fields": ("id,name,mimeType,size,modifiedTime,createdTime,webViewLink,parents,trashed"),
    }

    raw_response = await client.execute_tool_async(
        tool_slug=GOOGLEDRIVE_GET_FILE_ACTION,
        arguments=arguments,
    )

    data = _extract_dict(raw_response)
    if not data or ("id" not in data and "fileId" not in data):
        raise IntegrationExecutionError(
            f"Google Drive file '{clean_id}' not found or inaccessible.",
            integration_name="google_drive",
        )

    raw_size = data.get("size")
    size_int = int(raw_size) if raw_size and str(raw_size).isdigit() else None
    item_mime = data.get("mimeType") or data.get("mime_type") or "application/octet-stream"

    return DriveFileDetails(
        id=data.get("id") or clean_id,
        name=data.get("name") or data.get("title") or "Untitled",
        mime_type=item_mime,
        size=size_int,
        modified_time=data.get("modifiedTime") or data.get("modified_time"),
        created_time=data.get("createdTime") or data.get("created_time"),
        web_view_url=data.get("webViewLink") or data.get("web_view_url"),
        parents=data.get("parents") or [],
        is_folder=item_mime == GOOGLE_FOLDER_MIME_TYPE,
        trashed=data.get("trashed", False),
    )


async def download_drive_file(
    client: GoogleDriveClient,
    file_id: str,
    mime_type: str | None = None,
) -> DownloadDriveFileOutput:
    """Download file content from Google Drive, supporting Google Workspace exports.

    Args:
        client: Configured GoogleDriveClient instance.
        file_id: The unique identifier of the target file.
        mime_type: Optional requested export MIME type for Google Docs/Sheets.

    Returns:
        DownloadDriveFileOutput with decoded content or binary indicators.

    Raises:
        IntegrationExecutionError: If file is missing or download fails.
    """
    clean_id = file_id.strip()
    if not clean_id:
        raise IntegrationExecutionError(
            "File ID must not be empty.",
            integration_name="google_drive",
        )

    arguments: dict[str, Any] = {"fileId": clean_id}
    if mime_type:
        arguments["mime_type"] = mime_type.strip()

    raw_response = await client.execute_tool_async(
        tool_slug=GOOGLEDRIVE_DOWNLOAD_ACTION,
        arguments=arguments,
    )

    data = _extract_dict(raw_response)
    # Check for downloaded content keys across Composio download response variants
    content = data.get("downloaded_file_content") or data.get("content") or data.get("file_content")
    raw_size = data.get("size")
    size_int = int(raw_size) if raw_size and str(raw_size).isdigit() else len(str(content or ""))

    # Detect binary content
    is_binary = False
    encoding = "utf-8"
    if isinstance(content, bytes):
        try:
            content = content.decode("utf-8")
        except UnicodeDecodeError:
            is_binary = True
            content = None
            encoding = "binary"
    elif content is not None and not isinstance(content, str):
        content = str(content)

    return DownloadDriveFileOutput(
        file_id=clean_id,
        file_name=data.get("name") or data.get("file_name"),
        mime_type=data.get("mimeType") or mime_type or "text/plain",
        content=content,
        size=size_int,
        encoding=encoding,
        is_binary=is_binary,
    )


async def upload_drive_file(
    client: GoogleDriveClient,
    name: str,
    content: str,
    mime_type: str = "text/plain",
    parent_id: str | None = None,
) -> UploadDriveFileOutput:
    """Upload a new text or structured file to Google Drive.

    Args:
        client: Configured GoogleDriveClient instance.
        name: Name for the newly created file.
        content: Textual content of the file.
        mime_type: Target MIME type (default 'text/plain').
        parent_id: Optional folder ID to place the file inside.

    Returns:
        UploadDriveFileOutput with the created file metadata.

    Raises:
        IntegrationExecutionError: If name is blank or upload fails.
    """
    clean_name = name.strip()
    if not clean_name:
        raise IntegrationExecutionError(
            "File name must not be empty.",
            integration_name="google_drive",
        )

    arguments: dict[str, Any] = {
        "file_name": clean_name,
        "text_content": content,
        "mime_type": mime_type.strip() if mime_type else "text/plain",
    }
    if parent_id:
        arguments["parent_id"] = parent_id.strip()

    raw_response = await client.execute_tool_async(
        tool_slug=GOOGLEDRIVE_CREATE_TEXT_ACTION,
        arguments=arguments,
    )

    data = _extract_dict(raw_response)
    created_id = data.get("id") or data.get("file_id") or ""
    if not created_id:
        raise IntegrationExecutionError(
            f"Failed to retrieve created file ID for '{clean_name}'.",
            integration_name="google_drive",
        )

    return UploadDriveFileOutput(
        id=created_id,
        name=data.get("name") or clean_name,
        mime_type=data.get("mimeType") or mime_type,
        parent_id=parent_id,
        web_view_url=data.get("webViewLink") or data.get("web_view_url"),
    )


async def create_google_document(
    client: GoogleDriveClient,
    title: str,
    content: str = "",
    parent_id: str | None = None,
) -> CreateGoogleDocumentOutput:
    """Create a native Google Docs document with title and initial content.

    Args:
        client: Configured GoogleDriveClient instance.
        title: Title of the document.
        content: Initial plain text or Markdown content.
        parent_id: Optional folder ID where the document should be stored.

    Returns:
        CreateGoogleDocumentOutput containing document ID and web view link.

    Raises:
        IntegrationExecutionError: If title is empty or creation fails.
    """
    clean_title = title.strip()
    if not clean_title:
        raise IntegrationExecutionError(
            "Document title must not be empty.",
            integration_name="google_drive",
        )

    arguments: dict[str, Any] = {
        "file_name": clean_title,
        "text_content": content,
        "mime_type": GOOGLE_DOCS_MIME_TYPE,
    }
    if parent_id:
        arguments["parent_id"] = parent_id.strip()

    raw_response = await client.execute_tool_async(
        tool_slug=GOOGLEDRIVE_CREATE_TEXT_ACTION,
        arguments=arguments,
    )

    data = _extract_dict(raw_response)
    doc_id = data.get("id") or data.get("file_id") or ""
    if not doc_id:
        raise IntegrationExecutionError(
            f"Failed to create Google Document '{clean_title}'.",
            integration_name="google_drive",
        )

    return CreateGoogleDocumentOutput(
        document_id=doc_id,
        title=data.get("name") or clean_title,
        mime_type=GOOGLE_DOCS_MIME_TYPE,
        web_view_url=data.get("webViewLink") or data.get("web_view_url"),
    )


async def read_google_document(client: GoogleDriveClient, document_id: str) -> str:
    """Read plain text of a native Google Doc via Docs API or Google Drive."""
    clean_id = document_id.strip()
    if not clean_id:
        return ""

    try:
        data = await client.execute_tool_async(
            tool_slug=GOOGLEDOCS_GET_PLAINTEXT_ACTION,
            arguments={"document_id": clean_id},
        )
        dict_data = _extract_dict(data)
        if "text" in dict_data and dict_data["text"]:
            return str(dict_data["text"]).strip()
        if "content" in dict_data and dict_data["content"]:
            return str(dict_data["content"]).strip()
    except Exception as exc:
        logger.debug(
            "GOOGLEDOCS_GET_PLAINTEXT failed for '%s', using GOOGLEDRIVE_GET_DOCUMENT: %s",
            clean_id,
            exc,
        )

    try:
        doc_data = await client.execute_tool_async(
            tool_slug=GOOGLEDRIVE_GET_DOCUMENT_ACTION,
            arguments={"document_id": clean_id},
        )
        doc_dict = _extract_dict(doc_data)
        text_parts: list[str] = []
        body_content = doc_dict.get("body", {}).get("content", [])
        for elem in body_content:
            if "paragraph" in elem:
                for p_elem in elem["paragraph"].get("elements", []):
                    if "textRun" in p_elem:
                        text_parts.append(p_elem["textRun"].get("content", ""))
        return "".join(text_parts).strip()
    except Exception as exc:
        logger.debug("GOOGLEDRIVE_GET_DOCUMENT fallback for '%s' failed: %s", clean_id, exc)
        return ""


async def update_google_document(
    client: GoogleDriveClient,
    file_id: str,
    content: str,
    find_text: str | None = None,
    insertion_index: int | None = None,
    mime_type: str | None = None,
) -> UpdateGoogleDocumentOutput:
    """Update an existing Google Docs document or Drive text file safely.

    Detects the resource MIME type and routes:
    - Google Docs (application/vnd.google-apps.document) -> Google Docs API.
    - Standard text/binary files -> Google Drive API (GOOGLEDRIVE_EDIT_FILE).

    Args:
        client: Configured GoogleDriveClient instance.
        file_id: The ID of the file or document to update.
        content: New content to write, insert, or replace into the document.
        find_text: Optional text to find and replace across the Google Doc.
        insertion_index: Optional 1-based character position for insertion in a Google Doc.
        mime_type: Optional MIME type override. If omitted, discovered automatically.

    Returns:
        UpdateGoogleDocumentOutput with operation status, target type, and content preview.

    Raises:
        IntegrationExecutionError: If file ID is empty or update fails.
    """
    clean_id = file_id.strip()
    if not clean_id:
        raise IntegrationExecutionError(
            "File ID must not be empty.",
            integration_name="google_drive",
        )

    # Detect MIME type if not provided
    resolved_mime = mime_type.strip() if mime_type else None
    if not resolved_mime:
        try:
            file_meta = await get_drive_file(client, clean_id)
            resolved_mime = file_meta.mime_type
        except Exception as exc:
            logger.debug("Could not inspect metadata for '%s' prior to update: %s", clean_id, exc)
            resolved_mime = "text/plain"

    is_google_doc = resolved_mime == GOOGLE_DOCS_MIME_TYPE or "document" in (resolved_mime or "")

    if is_google_doc:
        target_type = "google_doc"
        if find_text:
            replace_args: dict[str, Any] = {
                "document_id": clean_id,
                "find_text": find_text,
                "replace_text": content,
                "match_case": False,
            }
            await client.execute_tool_async(
                tool_slug=GOOGLEDOCS_REPLACE_ALL_TEXT_ACTION,
                arguments=replace_args,
            )
            status = "replaced"
            msg = f"Successfully replaced occurrences of '{find_text}' in Google Doc '{clean_id}'."
        else:
            insert_args: dict[str, Any] = {
                "document_id": clean_id,
                "text_to_insert": content,
            }
            if insertion_index is not None and insertion_index >= 1:
                insert_args["insertion_index"] = insertion_index
                loc_desc = f"at index {insertion_index}"
            else:
                insert_args["append_to_end"] = True
                loc_desc = "at end of document"

            await client.execute_tool_async(
                tool_slug=GOOGLEDOCS_INSERT_TEXT_ACTION,
                arguments=insert_args,
            )
            status = "inserted"
            msg = f"Successfully inserted {len(content)} characters {loc_desc} in Doc '{clean_id}'."

        preview = await read_google_document(client, clean_id)
    else:
        target_type = "drive_file"
        arguments: dict[str, Any] = {
            "file_id": clean_id,
            "content": content,
            "mime_type": resolved_mime or "text/plain",
        }
        await client.execute_tool_async(
            tool_slug=GOOGLEDRIVE_EDIT_FILE_ACTION,
            arguments=arguments,
        )
        status = "updated"
        msg = f"Drive file '{clean_id}' successfully updated with {len(content)} characters."
        preview = content[:200] if len(content) > 200 else content

    return UpdateGoogleDocumentOutput(
        file_id=clean_id,
        status=status,
        message=msg,
        target_type=target_type,
        updated_content_preview=preview or None,
    )


async def create_drive_folder(
    client: GoogleDriveClient,
    name: str,
    parent_id: str | None = None,
) -> CreateDriveFolderOutput:
    """Create a new folder inside Google Drive, optionally under a specified parent folder.

    Args:
        client: Configured GoogleDriveClient instance.
        name: Name for the folder.
        parent_id: Optional ID of the parent folder.

    Returns:
        CreateDriveFolderOutput with created folder ID and link.

    Raises:
        IntegrationExecutionError: If folder name is empty or creation fails.
    """
    clean_name = name.strip()
    if not clean_name:
        raise IntegrationExecutionError(
            "Folder name must not be empty.",
            integration_name="google_drive",
        )

    arguments: dict[str, Any] = {"name": clean_name}
    if parent_id:
        arguments["parent_id"] = parent_id.strip()

    raw_response = await client.execute_tool_async(
        tool_slug=GOOGLEDRIVE_CREATE_FOLDER_ACTION,
        arguments=arguments,
    )

    data = _extract_dict(raw_response)
    folder_id = data.get("id") or data.get("folder_id") or ""
    if not folder_id:
        raise IntegrationExecutionError(
            f"Failed to create Google Drive folder '{clean_name}'.",
            integration_name="google_drive",
        )

    return CreateDriveFolderOutput(
        folder_id=folder_id,
        name=data.get("name") or clean_name,
        parent_id=parent_id,
        web_view_url=data.get("webViewLink") or data.get("web_view_url"),
    )
