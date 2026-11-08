"""Google Drive service integration package for SYNQRA.

Connects AI agents and MCP clients with Google Drive via Composio for file search,
metadata retrieval, content download/upload, Google Docs creation/updates, and folder management.
"""

from synqra.integrations.google_drive.client import GoogleDriveClient
from synqra.integrations.google_drive.integration import GoogleDriveIntegration
from synqra.integrations.google_drive.schemas import (
    CreateDriveFolderInput,
    CreateDriveFolderOutput,
    CreateGoogleDocumentInput,
    CreateGoogleDocumentOutput,
    DownloadDriveFileInput,
    DownloadDriveFileOutput,
    DriveFileDetails,
    GetDriveFileInput,
    SearchDriveFilesInput,
    SearchDriveFilesOutput,
    UpdateGoogleDocumentInput,
    UpdateGoogleDocumentOutput,
    UploadDriveFileInput,
    UploadDriveFileOutput,
)
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

__all__ = [
    "CreateDriveFolderInput",
    "CreateDriveFolderOutput",
    "CreateGoogleDocumentInput",
    "CreateGoogleDocumentOutput",
    "DownloadDriveFileInput",
    "DownloadDriveFileOutput",
    "DriveFileDetails",
    "GetDriveFileInput",
    "GoogleDriveClient",
    "GoogleDriveIntegration",
    "SearchDriveFilesInput",
    "SearchDriveFilesOutput",
    "UpdateGoogleDocumentInput",
    "UpdateGoogleDocumentOutput",
    "UploadDriveFileInput",
    "UploadDriveFileOutput",
    "create_drive_folder",
    "create_google_document",
    "download_drive_file",
    "get_drive_file",
    "read_google_document",
    "search_drive_files",
    "update_google_document",
    "upload_drive_file",
]
