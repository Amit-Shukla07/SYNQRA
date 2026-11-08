"""GitHub integration connector for SYNQRA.

Encapsulates GitHub repository exploration, metadata retrieval, file content access,
issue listing, and README extraction via Composio's session-based API.
"""

import logging

from fastmcp import FastMCP

from synqra.core.config import Settings, get_settings
from synqra.integrations.base import BaseIntegration
from synqra.integrations.github.client import GitHubClient
from synqra.integrations.github.schemas import (
    GitHubFileContentsOutput,
    GitHubListIssuesOutput,
    GitHubReadmeOutput,
    GitHubRepositoryDetails,
    GitHubSearchRepositoriesOutput,
)
from synqra.integrations.github.tools import (
    get_github_file_contents,
    get_github_readme,
    get_github_repository,
    list_github_issues,
    search_github_repositories,
)
from synqra.integrations.models import (
    IntegrationMetadata,
    IntegrationState,
    IntegrationStatus,
    utc_now,
)

logger = logging.getLogger("synqra.integrations.github")


class GitHubIntegration(BaseIntegration):
    """Production integration connector for GitHub via Composio."""

    def __init__(
        self,
        client: GitHubClient | None = None,
        settings: Settings | None = None,
        user_id: str | None = None,
    ) -> None:
        """Initialize the GitHub integration instance.

        Args:
            client: Optional pre-configured GitHubClient instance.
            settings: Optional application settings.
            user_id: Optional user identifier for Composio session scoping.
        """
        super().__init__(
            metadata=IntegrationMetadata(
                name="github",
                display_name="GitHub",
                description=(
                    "Search code repositories, inspect repository details, "
                    "retrieve file contents, list repository issues, and view README files."
                ),
                version="0.1.0",
                auth_required=True,
                auth_type="composio",
                documentation_url="https://docs.github.com/rest",
            )
        )
        self._settings = settings or get_settings()
        self.client = client or GitHubClient(
            settings=self._settings,
            user_id=user_id or self._settings.composio_user_id,
        )

    def is_configured(self) -> bool:
        """Check whether Composio credentials are configured."""
        if getattr(self.client, "is_configured", False):
            return True
        return self._settings.composio_api_key is not None

    async def test_connection(self) -> bool:
        """Verify that a valid, active GitHub connection exists through Composio."""
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
            logger.debug("Failed to obtain GitHub authorization URL: %s", exc)

        err_msg = (
            f"GitHub account not connected. Authorize at: {auth_url}"
            if auth_url
            else "GitHub account not connected. Authorize via Composio."
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
        """Register SYNQRA GitHub MCP tools onto the FastMCP server."""

        @server.tool(
            name="search_github_repositories",
            description=(
                "Search GitHub repositories by keywords, language, and sorting filters. "
                "Supports pagination and configurable result limits."
            ),
        )
        async def search_github_repositories_tool(
            query: str,
            language: str | None = None,
            sort: str = "stars",
            order: str = "desc",
            page: int = 1,
            per_page: int = 10,
        ) -> GitHubSearchRepositoriesOutput:
            """Search GitHub repositories by keywords and filters."""
            return await search_github_repositories(
                client=self.client,
                query=query,
                language=language,
                sort=sort,
                order=order,
                page=page,
                per_page=per_page,
            )

        @server.tool(
            name="get_github_repository",
            description=(
                "Retrieve detailed information and metrics for an individual GitHub repository, "
                "including description, stars, forks, language, topics, creation and update dates."
            ),
        )
        async def get_github_repository_tool(
            owner: str,
            repo: str,
        ) -> GitHubRepositoryDetails:
            """Retrieve repository details and metrics."""
            return await get_github_repository(
                client=self.client,
                owner=owner,
                repo=repo,
            )

        @server.tool(
            name="get_github_file_contents",
            description=(
                "Retrieve file content from a repository safely. Supports branch/tag/commit "
                "selection and handles binary and text files appropriately."
            ),
        )
        async def get_github_file_contents_tool(
            owner: str,
            repo: str,
            path: str,
            ref: str | None = None,
        ) -> GitHubFileContentsOutput:
            """Fetch file contents from a repository."""
            return await get_github_file_contents(
                client=self.client,
                owner=owner,
                repo=repo,
                path=path,
                ref=ref,
            )

        @server.tool(
            name="list_github_issues",
            description=(
                "Retrieve repository issues filtered by state and labels, strictly excluding "
                "pull requests. Supports pagination."
            ),
        )
        async def list_github_issues_tool(
            owner: str,
            repo: str,
            state: str = "open",
            labels: str | None = None,
            page: int = 1,
            per_page: int = 10,
        ) -> GitHubListIssuesOutput:
            """List repository issues excluding pull requests."""
            return await list_github_issues(
                client=self.client,
                owner=owner,
                repo=repo,
                state=state,
                labels=labels,
                page=page,
                per_page=per_page,
            )

        @server.tool(
            name="get_github_readme",
            description=("Retrieve the normalized README content and metadata for a repository."),
        )
        async def get_github_readme_tool(
            owner: str,
            repo: str,
            ref: str | None = None,
        ) -> GitHubReadmeOutput:
            """Fetch repository README content."""
            return await get_github_readme(
                client=self.client,
                owner=owner,
                repo=repo,
                ref=ref,
            )

        return [
            "search_github_repositories",
            "get_github_repository",
            "get_github_file_contents",
            "list_github_issues",
            "get_github_readme",
        ]
