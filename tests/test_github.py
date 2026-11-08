"""Unit and integration tests for SYNQRA GitHub integration.

Verifies GitHubClient, schema models, tools, and integration health checks using mocked
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
from synqra.integrations.github.client import (
    GITHUB_FIND_REPOSITORIES_ACTION,
    GitHubClient,
)
from synqra.integrations.github.integration import GitHubIntegration
from synqra.integrations.github.tools import (
    get_github_file_contents,
    get_github_readme,
    get_github_repository,
    list_github_issues,
    search_github_repositories,
)
from synqra.integrations.models import IntegrationState

# ==============================================================================
# Client & Configuration Tests
# ==============================================================================


def test_github_client_unconfigured_without_api_key() -> None:
    """Verify GitHubClient reports unconfigured when no Composio key is provided."""
    empty_settings = Settings()
    client = GitHubClient(settings=empty_settings)

    assert client.is_configured is False
    with pytest.raises(IntegrationConfigurationError) as exc_info:
        client._get_composio_client()
    assert "Composio API key is not configured" in str(exc_info.value)


def test_github_client_configured_with_api_key() -> None:
    """Verify GitHubClient reports configured when Composio key is present."""
    client = GitHubClient(api_key="ak_mock_key_12345")
    assert client.is_configured is True
    assert client.user_id == "synqra_default_user"


def test_github_client_check_connection_active() -> None:
    """Verify check_connection returns True when toolkit connection is active."""
    mock_session = MagicMock()
    mock_item = MagicMock()
    mock_item.slug = "github"
    mock_item.connection = MagicMock(is_active=True)
    mock_session.toolkits.return_value = MagicMock(items=[mock_item])

    client = GitHubClient(api_key="ak_mock_key", session=mock_session)
    assert client.check_connection() is True


def test_github_client_check_connection_inactive() -> None:
    """Verify check_connection returns False when toolkit connection is inactive."""
    mock_session = MagicMock()
    mock_item = MagicMock()
    mock_item.slug = "github"
    mock_item.connection = MagicMock(is_active=False)
    mock_session.toolkits.return_value = MagicMock(items=[mock_item])

    client = GitHubClient(api_key="ak_mock_key", session=mock_session)
    assert client.check_connection() is False


def test_github_client_authorize_generates_redirect_url() -> None:
    """Verify authorize returns redirect URL for OAuth connection."""
    mock_session = MagicMock()
    mock_session.authorize.return_value = MagicMock(
        redirect_url="https://connect.composio.dev/link/github_oauth_123"
    )

    client = GitHubClient(api_key="ak_mock_key", session=mock_session)
    url = client.authorize(callback_url="http://localhost:8000/callback")

    assert url == "https://connect.composio.dev/link/github_oauth_123"
    assert client.authorization_url == url


def test_github_client_execute_tool_success() -> None:
    """Verify execute_tool returns data dictionary on completed result."""
    mock_session = MagicMock()
    mock_session.execute.return_value = MagicMock(
        result_type="completed",
        error=None,
        data={"items": [{"name": "repo1"}]},
    )

    client = GitHubClient(api_key="ak_mock_key", session=mock_session)
    data = client.execute_tool(GITHUB_FIND_REPOSITORIES_ACTION, {"query": "python"})

    assert data == {"items": [{"name": "repo1"}]}


def test_github_client_execute_tool_rate_limit_error() -> None:
    """Verify execute_tool identifies rate limits and raises IntegrationExecutionError."""
    mock_session = MagicMock()
    mock_session.execute.return_value = MagicMock(
        result_type="failed",
        error="API rate limit exceeded for user",
        data=None,
    )

    client = GitHubClient(api_key="ak_mock_key", session=mock_session)
    with pytest.raises(IntegrationExecutionError) as exc_info:
        client.execute_tool(GITHUB_FIND_REPOSITORIES_ACTION, {"query": "test"})
    assert "rate limit" in str(exc_info.value).lower()


# ==============================================================================
# Tool 1: search_github_repositories Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_search_github_repositories_success() -> None:
    """Verify repository search parameters, normalization, and pagination."""
    mock_client = MagicMock(spec=GitHubClient)
    mock_client.execute_tool_async.return_value = {
        "total_count": 1,
        "items": [
            {
                "name": "synqra-core",
                "full_name": "synqra/synqra-core",
                "owner": {"login": "synqra"},
                "description": "Universal integration server",
                "html_url": "https://github.com/synqra/synqra-core",
                "stargazers_count": 150,
                "forks_count": 25,
                "language": "Python",
                "topics": ["mcp", "fastmcp"],
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-02-01T00:00:00Z",
            }
        ],
    }

    result = await search_github_repositories(
        client=mock_client,
        query="synqra",
        language="python",
        sort="stars",
        order="desc",
        page=1,
        per_page=10,
    )

    assert result.query == "synqra"
    assert result.total_count == 1
    assert len(result.repositories) == 1

    repo = result.repositories[0]
    assert repo.full_name == "synqra/synqra-core"
    assert repo.owner == "synqra"
    assert repo.stars == 150
    assert repo.forks == 25
    assert repo.language == "Python"
    assert "mcp" in repo.topics


@pytest.mark.asyncio
async def test_search_github_repositories_empty_query() -> None:
    """Verify empty search query returns empty result without invoking API."""
    mock_client = MagicMock(spec=GitHubClient)
    result = await search_github_repositories(client=mock_client, query="   ")
    assert result.total_count == 0
    assert result.repositories == []
    mock_client.execute_tool_async.assert_not_called()


# ==============================================================================
# Tool 2: get_github_repository Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_get_github_repository_success() -> None:
    """Verify repository details retrieval and normalization."""
    mock_client = MagicMock(spec=GitHubClient)
    mock_client.execute_tool_async.return_value = {
        "name": "requests",
        "full_name": "psf/requests",
        "owner": {"login": "psf"},
        "description": "A simple, yet elegant, HTTP library.",
        "html_url": "https://github.com/psf/requests",
        "stargazers_count": 51000,
        "forks_count": 9200,
        "language": "Python",
        "topics": ["http", "python", "requests"],
        "default_branch": "main",
        "open_issues_count": 180,
        "license": {"spdx_id": "Apache-2.0"},
        "created_at": "2011-02-13T18:38:17Z",
        "updated_at": "2026-03-01T10:00:00Z",
    }

    details = await get_github_repository(client=mock_client, owner="psf", repo="requests")

    assert details.full_name == "psf/requests"
    assert details.owner == "psf"
    assert details.stars == 51000
    assert details.forks == 9200
    assert details.language == "Python"
    assert details.license_name == "Apache-2.0"
    assert details.default_branch == "main"
    assert "http" in details.topics


@pytest.mark.asyncio
async def test_get_github_repository_validation_and_missing() -> None:
    """Verify error raised on empty parameters or missing repository."""
    mock_client = MagicMock(spec=GitHubClient)

    # Empty owner
    with pytest.raises(IntegrationExecutionError):
        await get_github_repository(client=mock_client, owner="", repo="repo")

    # Missing repo
    mock_client.execute_tool_async.return_value = {}
    with pytest.raises(IntegrationExecutionError) as exc_info:
        await get_github_repository(client=mock_client, owner="owner", repo="nonexistent")
    assert "not found" in str(exc_info.value)


# ==============================================================================
# Tool 3: get_github_file_contents Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_get_github_file_contents_text_decoding() -> None:
    """Verify text file base64 decoding."""
    import base64

    text_content = "def hello_world():\n    return 'Hello, World!'\n"
    encoded = base64.b64encode(text_content.encode("utf-8")).decode("utf-8")

    mock_client = MagicMock(spec=GitHubClient)
    mock_client.execute_tool_async.return_value = {
        "size": len(text_content),
        "encoding": "base64",
        "content": encoded,
        "download_url": "https://raw.githubusercontent.com/owner/repo/main/hello.py",
    }

    result = await get_github_file_contents(
        client=mock_client, owner="owner", repo="repo", path="hello.py", ref="main"
    )

    assert result.content == text_content
    assert result.encoding == "utf-8"
    assert result.is_binary is False
    assert result.size == len(text_content)


@pytest.mark.asyncio
async def test_get_github_file_contents_binary_detection() -> None:
    """Verify binary file containing null bytes is detected and marked binary."""
    import base64

    binary_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    encoded = base64.b64encode(binary_bytes).decode("utf-8")

    mock_client = MagicMock(spec=GitHubClient)
    mock_client.execute_tool_async.return_value = {
        "size": len(binary_bytes),
        "encoding": "base64",
        "content": encoded,
    }

    result = await get_github_file_contents(
        client=mock_client, owner="owner", repo="repo", path="logo.png"
    )

    assert result.is_binary is True
    assert result.encoding == "binary"
    assert result.content is None


@pytest.mark.asyncio
async def test_get_github_file_contents_large_file_safety() -> None:
    """Verify file exceeding size limit is handled safely without memory explosion."""
    mock_client = MagicMock(spec=GitHubClient)
    mock_client.execute_tool_async.return_value = {
        "size": 5 * 1024 * 1024,  # 5 MB
        "download_url": "https://raw.githubusercontent.com/owner/repo/main/dataset.csv",
    }

    result = await get_github_file_contents(
        client=mock_client, owner="owner", repo="repo", path="dataset.csv"
    )

    assert result.is_binary is True
    assert result.content is None
    assert result.size == 5 * 1024 * 1024
    assert result.download_url is not None


# ==============================================================================
# Tool 4: list_github_issues Tests (PR Exclusion Critical Requirement)
# ==============================================================================


@pytest.mark.asyncio
async def test_list_github_issues_excludes_pull_requests() -> None:
    """CRITICAL REQUIREMENT: Verify pull requests are strictly excluded from issue results."""
    mock_client = MagicMock(spec=GitHubClient)
    mock_client.execute_tool_async.return_value = {
        "items": [
            {
                "number": 101,
                "title": "Bug: memory leak in worker",
                "state": "open",
                "html_url": "https://github.com/owner/repo/issues/101",
                "user": {"login": "alice"},
                "labels": [{"name": "bug"}, {"name": "priority"}],
                "comments": 4,
                "body": "There is a memory leak when processing large payloads.",
                "created_at": "2026-03-01T00:00:00Z",
                # Pure issue: no 'pull_request' key
            },
            {
                "number": 102,
                "title": "PR: fix memory leak",
                "state": "open",
                "html_url": "https://github.com/owner/repo/pull/102",
                "user": {"login": "bob"},
                "pull_request": {"url": "https://api.github.com/repos/owner/repo/pulls/102"},
                # This is a pull request and MUST be excluded
            },
            {
                "number": 103,
                "title": "Feature: add caching layer",
                "state": "open",
                "html_url": "https://github.com/owner/repo/issues/103",
                "user": {"login": "carol"},
                "labels": [{"name": "enhancement"}],
                "comments": 1,
                "body": "Proposing a Redis-based cache layer.",
                "created_at": "2026-03-02T00:00:00Z",
            },
        ]
    }

    result = await list_github_issues(client=mock_client, owner="owner", repo="repo", state="open")

    # PR #102 must be excluded! Only issues #101 and #103 should remain
    assert result.total_returned == 2
    issue_numbers = [issue.number for issue in result.issues]
    assert issue_numbers == [101, 103]
    assert 102 not in issue_numbers

    # Verify first issue normalization
    issue1 = result.issues[0]
    assert issue1.title == "Bug: memory leak in worker"
    assert issue1.author == "alice"
    assert "bug" in issue1.labels
    assert issue1.comments_count == 4


# ==============================================================================
# Tool 5: get_github_readme Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_get_github_readme_success() -> None:
    """Verify README content decoding and metadata retrieval."""
    import base64

    readme_markdown = "# SYNQRA\n\nUniversal integration MCP server.\n"
    encoded = base64.b64encode(readme_markdown.encode("utf-8")).decode("utf-8")

    mock_client = MagicMock(spec=GitHubClient)
    mock_client.execute_tool_async.return_value = {
        "name": "README.md",
        "size": len(readme_markdown),
        "encoding": "base64",
        "content": encoded,
        "download_url": "https://raw.githubusercontent.com/owner/repo/main/README.md",
    }

    result = await get_github_readme(client=mock_client, owner="owner", repo="repo")

    assert result.filename == "README.md"
    assert result.content == readme_markdown
    assert result.encoding == "utf-8"
    assert result.size == len(readme_markdown)


@pytest.mark.asyncio
async def test_get_github_readme_missing() -> None:
    """Verify error raised when repository has no README."""
    mock_client = MagicMock(spec=GitHubClient)
    mock_client.execute_tool_async.return_value = {}

    with pytest.raises(IntegrationExecutionError) as exc_info:
        await get_github_readme(client=mock_client, owner="owner", repo="noreadme")
    assert "README not found" in str(exc_info.value)


# ==============================================================================
# Integration Lifecycle & Health Check Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_github_integration_health_check_flow() -> None:
    """Verify GitHubIntegration health check states: UNCONFIGURED, CONFIGURED, CONNECTED."""
    # 1. Unconfigured
    unconf_client = MagicMock(spec=GitHubClient)
    unconf_client.is_configured = False
    unconf_integration = GitHubIntegration(client=unconf_client)
    status_unconf = await unconf_integration.health_check()
    assert status_unconf.state == IntegrationState.UNCONFIGURED
    assert status_unconf.is_connected is False

    # 2. Configured but disconnected (returns OAuth authorization link)
    auth_client = MagicMock(spec=GitHubClient)
    auth_client.is_configured = True
    auth_client.check_connection_async.return_value = False
    auth_client.authorize_async.return_value = "https://connect.composio.dev/link/github_oauth_xyz"
    auth_integration = GitHubIntegration(client=auth_client)
    status_auth = await auth_integration.health_check()
    assert status_auth.state == IntegrationState.CONFIGURED
    assert status_auth.is_connected is False
    assert "https://connect.composio.dev/link/github_oauth_xyz" in str(status_auth.error_message)

    # 3. Connected
    conn_client = MagicMock(spec=GitHubClient)
    conn_client.is_configured = True
    conn_client.check_connection_async.return_value = True
    conn_integration = GitHubIntegration(client=conn_client)
    status_conn = await conn_integration.health_check()
    assert status_conn.state == IntegrationState.CONNECTED
    assert status_conn.is_connected is True


@pytest.mark.asyncio
async def test_github_integration_tool_registration() -> None:
    """Verify GitHubIntegration registers all 5 specified MCP tools onto FastMCP."""
    mock_client = MagicMock(spec=GitHubClient)
    mock_client.is_configured = True
    mock_client.check_connection_async.return_value = True

    server = FastMCP("test-server")
    integration = GitHubIntegration(client=mock_client)
    registered = integration.register_tools(server)

    assert len(registered) == 5
    assert "search_github_repositories" in registered
    assert "get_github_repository" in registered
    assert "get_github_file_contents" in registered
    assert "list_github_issues" in registered
    assert "get_github_readme" in registered
