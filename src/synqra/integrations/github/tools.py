"""Tool implementations for the GitHub integration in SYNQRA.

Provides sanitized, well-structured tools for repository search, metadata inspection,
file content retrieval, issue listing, and README extraction via Composio.
"""

import base64
import logging
from typing import Any

from synqra.integrations.exceptions import IntegrationExecutionError
from synqra.integrations.github.client import (
    GITHUB_FIND_REPOSITORIES_ACTION,
    GITHUB_GET_CONTENT_ACTION,
    GITHUB_GET_README_ACTION,
    GITHUB_GET_REPOSITORY_ACTION,
    GITHUB_LIST_ISSUES_ACTION,
    GitHubClient,
)
from synqra.integrations.github.schemas import (
    GitHubFileContentsOutput,
    GitHubIssueItem,
    GitHubListIssuesOutput,
    GitHubReadmeOutput,
    GitHubRepositoryDetails,
    GitHubRepositorySummary,
    GitHubSearchRepositoriesOutput,
)

logger = logging.getLogger("synqra.integrations.github.tools")

MAX_TEXT_FILE_SIZE_BYTES = 1024 * 1024  # 1MB limit for in-memory text decoding


def _extract_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract item records from various Composio response envelope structures."""
    if not isinstance(payload, dict):
        return []

    # Direct items list (standard GitHub search response)
    if "items" in payload and isinstance(payload["items"], list):
        return payload["items"]

    # Nested data.items
    data = payload.get("data")
    if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
        return data["items"]

    # Direct list in data
    if isinstance(data, list):
        return data

    # Direct list in payload results
    results = payload.get("results")
    if isinstance(results, list):
        return results

    # Direct repositories or issues list
    for key in ("repositories", "issues"):
        if key in payload and isinstance(payload[key], list):
            return payload[key]
        if isinstance(data, dict) and key in data and isinstance(data[key], list):
            return data[key]

    return []


def _extract_dict(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract a dictionary payload from various Composio envelope structures."""
    if not isinstance(payload, dict):
        return {}
    if "data" in payload and isinstance(payload["data"], dict):
        return payload["data"]
    return payload


def _decode_content(
    raw_content: str | None, raw_encoding: str | None
) -> tuple[str | None, str, bool]:
    """Safely decode GitHub file or README content from base64 representation.

    Returns:
        tuple of (decoded_text, encoding_name, is_binary)
    """
    if not raw_content:
        return "", "utf-8", False

    # Check encoding type
    if (raw_encoding or "").lower() == "base64" or "\n" in raw_content:
        try:
            # GitHub Base64 content often contains newlines
            cleaned = raw_content.replace("\n", "").replace("\r", "")
            decoded_bytes = base64.b64decode(cleaned)

            # Check if file is binary (contains null bytes)
            if b"\x00" in decoded_bytes:
                return None, "binary", True

            # Attempt UTF-8 decoding
            text = decoded_bytes.decode("utf-8", errors="replace")
            return text, "utf-8", False
        except Exception as exc:
            logger.debug("Failed to decode base64 content: %s", exc)
            return None, "binary", True

    return raw_content, "utf-8", False


async def search_github_repositories(
    client: GitHubClient,
    query: str,
    language: str | None = None,
    sort: str | None = "stars",
    order: str = "desc",
    page: int = 1,
    per_page: int = 10,
) -> GitHubSearchRepositoriesOutput:
    """Search GitHub repositories by keywords, language, and sorting filters.

    Args:
        client: Configured GitHubClient instance.
        query: Search keywords or query string.
        language: Optional language filter (e.g. 'python', 'rust').
        sort: Field to sort by ('stars', 'forks', 'updated').
        order: Sort order ('asc', 'desc').
        page: Page number (1-based).
        per_page: Number of repositories to return per page (1-100).

    Returns:
        GitHubSearchRepositoriesOutput containing matching repository summaries.
    """
    clean_query = query.strip()
    if not clean_query:
        return GitHubSearchRepositoriesOutput(
            query=query,
            total_count=0,
            page=page,
            repositories=[],
        )

    limit = max(1, min(per_page, 100))
    current_page = max(1, page)

    arguments: dict[str, Any] = {
        "query": clean_query,
        "page": current_page,
        "per_page": limit,
        "order": order,
    }
    if language:
        arguments["language"] = language.strip().lower()
    if sort:
        arguments["sort"] = sort

    raw_response = await client.execute_tool_async(
        tool_slug=GITHUB_FIND_REPOSITORIES_ACTION,
        arguments=arguments,
    )

    items = _extract_items(raw_response)
    total_count = raw_response.get("total_count")
    if total_count is None and isinstance(raw_response.get("data"), dict):
        total_count = raw_response["data"].get("total_count", len(items))

    repo_items: list[GitHubRepositorySummary] = []
    for item in items:
        owner_obj = item.get("owner", {})
        owner_login = (
            owner_obj.get("login") if isinstance(owner_obj, dict) else str(owner_obj or "")
        )
        full_name = item.get("full_name") or f"{owner_login}/{item.get('name', '')}"

        repo_items.append(
            GitHubRepositorySummary(
                full_name=full_name,
                name=item.get("name", ""),
                owner=owner_login,
                description=item.get("description"),
                url=item.get("html_url") or f"https://github.com/{full_name}",
                stars=item.get("stargazers_count", 0),
                forks=item.get("forks_count", 0),
                language=item.get("language"),
                topics=item.get("topics", []),
                created_at=item.get("created_at"),
                updated_at=item.get("updated_at"),
            )
        )

    return GitHubSearchRepositoriesOutput(
        query=clean_query,
        total_count=total_count or len(repo_items),
        page=current_page,
        repositories=repo_items,
    )


async def get_github_repository(
    client: GitHubClient,
    owner: str,
    repo: str,
) -> GitHubRepositoryDetails:
    """Retrieve detailed information and metrics for an individual GitHub repository.

    Args:
        client: Configured GitHubClient instance.
        owner: GitHub account owner username or organization.
        repo: Repository name.

    Returns:
        GitHubRepositoryDetails model with repository metadata.

    Raises:
        IntegrationExecutionError: If repository is missing or inaccessible.
    """
    clean_owner = owner.strip()
    clean_repo = repo.strip()
    if not clean_owner or not clean_repo:
        raise IntegrationExecutionError(
            "Both 'owner' and 'repo' are required.",
            integration_name="github",
        )

    arguments = {
        "owner": clean_owner,
        "repo": clean_repo,
    }

    raw_response = await client.execute_tool_async(
        tool_slug=GITHUB_GET_REPOSITORY_ACTION,
        arguments=arguments,
    )

    data = _extract_dict(raw_response)
    if not data or ("name" not in data and "full_name" not in data):
        raise IntegrationExecutionError(
            f"Repository '{clean_owner}/{clean_repo}' not found or inaccessible.",
            integration_name="github",
        )

    owner_obj = data.get("owner", {})
    owner_login = owner_obj.get("login") if isinstance(owner_obj, dict) else clean_owner
    full_name = data.get("full_name") or f"{clean_owner}/{clean_repo}"

    license_obj = data.get("license")
    license_name = (
        license_obj.get("spdx_id") or license_obj.get("name")
        if isinstance(license_obj, dict)
        else None
    )

    return GitHubRepositoryDetails(
        full_name=full_name,
        name=data.get("name", clean_repo),
        owner=owner_login,
        description=data.get("description"),
        url=data.get("html_url") or f"https://github.com/{full_name}",
        stars=data.get("stargazers_count", 0),
        forks=data.get("forks_count", 0),
        language=data.get("language"),
        topics=data.get("topics", []),
        created_at=data.get("created_at"),
        updated_at=data.get("updated_at"),
        default_branch=data.get("default_branch", "main"),
        open_issues_count=data.get("open_issues_count", 0),
        license_name=license_name,
    )


async def get_github_file_contents(
    client: GitHubClient,
    owner: str,
    repo: str,
    path: str,
    ref: str | None = None,
) -> GitHubFileContentsOutput:
    """Retrieve file content from a repository safely handling binary and text formats.

    Args:
        client: Configured GitHubClient instance.
        owner: Repository owner.
        repo: Repository name.
        path: Path to the target file.
        ref: Optional branch name, tag, or commit SHA.

    Returns:
        GitHubFileContentsOutput with decoded content and metadata.

    Raises:
        IntegrationExecutionError: If file is missing or invalid.
    """
    clean_owner = owner.strip()
    clean_repo = repo.strip()
    clean_path = path.strip().lstrip("/")

    if not clean_owner or not clean_repo or not clean_path:
        raise IntegrationExecutionError(
            "'owner', 'repo', and 'path' are all required.",
            integration_name="github",
        )

    arguments: dict[str, Any] = {
        "owner": clean_owner,
        "repo": clean_repo,
        "path": clean_path,
    }
    if ref:
        arguments["ref"] = ref.strip()

    raw_response = await client.execute_tool_async(
        tool_slug=GITHUB_GET_CONTENT_ACTION,
        arguments=arguments,
    )

    data = _extract_dict(raw_response)
    if isinstance(data.get("content"), dict):
        data = data["content"]

    if not data or ("content" not in data and "size" not in data):
        raise IntegrationExecutionError(
            f"File '{clean_path}' not found in repository '{clean_owner}/{clean_repo}'.",
            integration_name="github",
        )

    file_size = data.get("size", 0)
    raw_content = data.get("content", "")
    raw_encoding = data.get("encoding", "base64")

    # Safe handling of large files
    if file_size > MAX_TEXT_FILE_SIZE_BYTES:
        return GitHubFileContentsOutput(
            owner=clean_owner,
            repo=clean_repo,
            path=clean_path,
            ref=ref,
            content=None,
            size=file_size,
            encoding="binary",
            is_binary=True,
            download_url=data.get("download_url"),
        )

    decoded_text, encoding, is_binary = _decode_content(raw_content, raw_encoding)

    return GitHubFileContentsOutput(
        owner=clean_owner,
        repo=clean_repo,
        path=clean_path,
        ref=ref,
        content=decoded_text,
        size=file_size,
        encoding=encoding,
        is_binary=is_binary,
        download_url=data.get("download_url"),
    )


async def list_github_issues(
    client: GitHubClient,
    owner: str,
    repo: str,
    state: str = "open",
    labels: str | None = None,
    page: int = 1,
    per_page: int = 10,
) -> GitHubListIssuesOutput:
    """Retrieve repository issues, strictly excluding pull requests.

    Args:
        client: Configured GitHubClient instance.
        owner: Repository owner.
        repo: Repository name.
        state: Filter by state ('open', 'closed', 'all').
        labels: Optional comma-separated label list.
        page: Page number (1-based).
        per_page: Number of issues per page (1-100).

    Returns:
        GitHubListIssuesOutput with filtered issue list.
    """
    clean_owner = owner.strip()
    clean_repo = repo.strip()
    if not clean_owner or not clean_repo:
        raise IntegrationExecutionError(
            "Both 'owner' and 'repo' are required.",
            integration_name="github",
        )

    current_page = max(1, page)
    limit = max(1, min(per_page, 100))

    arguments: dict[str, Any] = {
        "owner": clean_owner,
        "repo": clean_repo,
        "state": state,
        "page": current_page,
        "per_page": limit,
    }
    if labels:
        arguments["labels"] = labels.strip()

    raw_response = await client.execute_tool_async(
        tool_slug=GITHUB_LIST_ISSUES_ACTION,
        arguments=arguments,
    )

    items = _extract_items(raw_response)
    issues_list: list[GitHubIssueItem] = []

    for item in items:
        # Exclude Pull Requests: in GitHub's API, pull requests have a 'pull_request' key
        if "pull_request" in item:
            continue

        author_obj = item.get("user", {})
        author_login = author_obj.get("login") if isinstance(author_obj, dict) else None

        label_objs = item.get("labels", [])
        label_names = [lbl.get("name") if isinstance(lbl, dict) else str(lbl) for lbl in label_objs]

        body = item.get("body") or ""
        preview = body[:200] + "..." if len(body) > 200 else body

        issues_list.append(
            GitHubIssueItem(
                number=item.get("number", 0),
                title=item.get("title", "Untitled Issue"),
                state=item.get("state", "open"),
                url=item.get("html_url")
                or f"https://github.com/{clean_owner}/{clean_repo}/issues/{item.get('number')}",
                author=author_login,
                labels=label_names,
                created_at=item.get("created_at"),
                updated_at=item.get("updated_at"),
                comments_count=item.get("comments", 0),
                body_preview=preview or None,
            )
        )

    return GitHubListIssuesOutput(
        owner=clean_owner,
        repo=clean_repo,
        page=current_page,
        total_returned=len(issues_list),
        issues=issues_list,
    )


async def get_github_readme(
    client: GitHubClient,
    owner: str,
    repo: str,
    ref: str | None = None,
) -> GitHubReadmeOutput:
    """Retrieve the normalized README content for a repository.

    Args:
        client: Configured GitHubClient instance.
        owner: Repository owner.
        repo: Repository name.
        ref: Optional branch or commit reference.

    Returns:
        GitHubReadmeOutput with decoded text content and metadata.

    Raises:
        IntegrationExecutionError: If README is missing or request fails.
    """
    clean_owner = owner.strip()
    clean_repo = repo.strip()
    if not clean_owner or not clean_repo:
        raise IntegrationExecutionError(
            "Both 'owner' and 'repo' are required.",
            integration_name="github",
        )

    arguments: dict[str, Any] = {
        "owner": clean_owner,
        "repo": clean_repo,
    }
    if ref:
        arguments["ref"] = ref.strip()

    raw_response = await client.execute_tool_async(
        tool_slug=GITHUB_GET_README_ACTION,
        arguments=arguments,
    )

    data = _extract_dict(raw_response)
    if isinstance(data.get("content"), dict):
        data = data["content"]

    if not data or "content" not in data:
        raise IntegrationExecutionError(
            f"README not found in repository '{clean_owner}/{clean_repo}'.",
            integration_name="github",
        )

    raw_content = data.get("content", "")
    raw_encoding = data.get("encoding", "base64")
    decoded_text, encoding, _ = _decode_content(raw_content, raw_encoding)

    return GitHubReadmeOutput(
        owner=clean_owner,
        repo=clean_repo,
        ref=ref,
        filename=data.get("name", "README.md"),
        content=decoded_text or "",
        size=data.get("size", len(decoded_text or "")),
        encoding=encoding,
        download_url=data.get("download_url"),
    )
