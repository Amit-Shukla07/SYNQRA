"""Pydantic schemas for the Google Drive integration in SYNQRA.

Defines validated input parameters and structured response models for file search,
metadata retrieval, downloads, uploads, folder creation, and Google Document operations.
"""

from pydantic import BaseModel, Field

GOOGLE_DOCS_MIME_TYPE = "application/vnd.google-apps.document"
GOOGLE_FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"


class DriveFileSummary(BaseModel):
    """Normalized summary of a file or folder in Google Drive."""

    id: str = Field(description="Google Drive file or folder ID.")
    name: str = Field(description="Name of the file or folder.")
    mime_type: str = Field(description="MIME type of the resource.")
    size: int | None = Field(
        default=None, description="File size in bytes (None for folders/docs)."
    )
    modified_time: str | None = Field(
        default=None, description="ISO timestamp of last modification."
    )
    web_view_url: str | None = Field(
        default=None, description="URL to view the item in Google Drive."
    )
    is_folder: bool = Field(
        default=False, description="Flag indicating if the resource is a folder."
    )


class SearchDriveFilesInput(BaseModel):
    """Input parameters for searching files and folders in Google Drive."""

    query: str | None = Field(
        default=None,
        description="Text search query or Google Drive filter (e.g. 'quarterly report').",
    )
    name: str | None = Field(
        default=None,
        description="Filter files containing this specific name.",
    )
    mime_type: str | None = Field(
        default=None,
        description="Filter by MIME type (e.g. 'application/pdf', 'text/plain').",
    )
    folder_id: str | None = Field(
        default=None,
        description="Search exclusively inside this parent folder ID.",
    )
    page_size: int = Field(default=10, ge=1, le=100, description="Maximum files to return (1-100).")


class SearchDriveFilesOutput(BaseModel):
    """Structured response containing Google Drive file search results."""

    total_count: int = Field(description="Number of items returned in this result set.")
    files: list[DriveFileSummary] = Field(
        default_factory=list, description="Matching files and folders."
    )


class GetDriveFileInput(BaseModel):
    """Input parameters for fetching metadata of an individual Google Drive file."""

    file_id: str = Field(description="Google Drive unique file ID.", min_length=1)


class DriveFileDetails(BaseModel):
    """Detailed metadata for a Google Drive file or folder."""

    id: str = Field(description="File ID.")
    name: str = Field(description="File name.")
    mime_type: str = Field(description="Resource MIME type.")
    size: int | None = Field(default=None, description="Size in bytes.")
    modified_time: str | None = Field(default=None, description="Last modification timestamp.")
    created_time: str | None = Field(default=None, description="Creation timestamp.")
    web_view_url: str | None = Field(
        default=None, description="Web link to view the file in browser."
    )
    parents: list[str] = Field(default_factory=list, description="List of parent folder IDs.")
    is_folder: bool = Field(default=False, description="Whether the item is a folder.")
    trashed: bool = Field(default=False, description="Whether the file is in the trash.")


class DownloadDriveFileInput(BaseModel):
    """Input parameters for downloading file content from Google Drive."""

    file_id: str = Field(description="Google Drive file ID.", min_length=1)
    mime_type: str | None = Field(
        default=None,
        description=(
            "MIME type for Google Docs export (e.g. 'text/plain', 'application/pdf'). "
            "Leave blank for native binary or text files."
        ),
    )


class DownloadDriveFileOutput(BaseModel):
    """Normalized response containing downloaded file content."""

    file_id: str = Field(description="File ID.")
    file_name: str | None = Field(default=None, description="File name if available.")
    mime_type: str | None = Field(default=None, description="MIME type of the content.")
    content: str | None = Field(default=None, description="Text content (or None if binary).")
    size: int | None = Field(default=None, description="Size in bytes.")
    encoding: str = Field(default="utf-8", description="Encoding ('utf-8', 'base64', 'binary').")
    is_binary: bool = Field(default=False, description="True if the file is binary or non-text.")


class UploadDriveFileInput(BaseModel):
    """Input parameters for uploading a file to Google Drive."""

    name: str = Field(description="Desired file name on Google Drive.", min_length=1)
    content: str = Field(description="Text content to be stored in the file.")
    mime_type: str = Field(default="text/plain", description="MIME type for the uploaded file.")
    parent_id: str | None = Field(default=None, description="Optional target folder ID.")


class UploadDriveFileOutput(BaseModel):
    """Structured response for uploaded Google Drive file."""

    id: str = Field(description="Newly created Google Drive file ID.")
    name: str = Field(description="File name.")
    mime_type: str = Field(description="File MIME type.")
    parent_id: str | None = Field(default=None, description="Parent folder ID.")
    web_view_url: str | None = Field(default=None, description="Link to view the uploaded file.")


class CreateGoogleDocumentInput(BaseModel):
    """Input parameters for creating a native Google Docs document."""

    title: str = Field(description="Title of the new Google Document.", min_length=1)
    content: str = Field(default="", description="Initial text content for the document.")
    parent_id: str | None = Field(default=None, description="Optional parent folder ID.")


class CreateGoogleDocumentOutput(BaseModel):
    """Structured response for created Google Document."""

    document_id: str = Field(description="Google Docs document ID.")
    title: str = Field(description="Document title.")
    mime_type: str = Field(default=GOOGLE_DOCS_MIME_TYPE, description="Google Docs MIME type.")
    web_view_url: str | None = Field(default=None, description="Web link to open the document.")


class UpdateGoogleDocumentInput(BaseModel):
    """Input parameters for updating a Google Document or Drive file."""

    file_id: str = Field(description="Google Document or Drive file ID to update.", min_length=1)
    content: str = Field(description="New content to write, insert, or replace into the document.")
    find_text: str | None = Field(
        default=None,
        description="If provided for a Google Doc, replaces find_text with content.",
    )
    insertion_index: int | None = Field(
        default=None,
        description="1-based character position for insertion. Defaults to appending.",
    )
    mime_type: str | None = Field(
        default=None,
        description="Optional MIME type override (e.g. 'text/plain').",
    )


class UpdateGoogleDocumentOutput(BaseModel):
    """Structured response for updated Google Document or Drive file."""

    file_id: str = Field(description="File or document ID that was updated.")
    status: str = Field(
        default="updated", description="Operation status ('updated', 'inserted', 'replaced')."
    )
    message: str = Field(description="Summary message.")
    target_type: str = Field(
        default="google_doc",
        description="Resource type updated: 'google_doc' or 'drive_file'.",
    )
    updated_content_preview: str | None = Field(
        default=None,
        description="Text content preview read back from the document or file.",
    )


class CreateDriveFolderInput(BaseModel):
    """Input parameters for creating a new Google Drive folder."""

    name: str = Field(description="Name of the new folder.", min_length=1)
    parent_id: str | None = Field(default=None, description="Optional parent folder ID.")


class CreateDriveFolderOutput(BaseModel):
    """Structured response for created Google Drive folder."""

    folder_id: str = Field(description="Created folder ID.")
    name: str = Field(description="Folder name.")
    parent_id: str | None = Field(default=None, description="Parent folder ID.")
    web_view_url: str | None = Field(default=None, description="Link to open folder in Drive.")
