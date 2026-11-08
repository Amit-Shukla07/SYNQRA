"""Web Search integration connector for SYNQRA.

Encapsulates web searching, news retrieval, and webpage content extraction
via the Tavily API, providing complete error isolation and SSRF protection.
"""

import logging

from fastmcp import FastMCP

from synqra.core.config import Settings, get_settings
from synqra.integrations.base import BaseIntegration
from synqra.integrations.models import (
    IntegrationMetadata,
    IntegrationState,
    IntegrationStatus,
    utc_now,
)
from synqra.integrations.web_search.client import WebSearchClient
from synqra.integrations.web_search.schemas import (
    ExtractWebpageOutput,
    SearchNewsOutput,
    SearchWebOutput,
)
from synqra.integrations.web_search.tools import (
    extract_webpage_content,
    search_news,
    search_web,
)

logger = logging.getLogger("synqra.integrations.web_search")


class WebSearchIntegration(BaseIntegration):
    """Production integration connector for Web Search via Tavily."""

    def __init__(
        self,
        client: WebSearchClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the Web Search integration instance.

        Args:
            client: Optional pre-configured WebSearchClient instance.
            settings: Optional application settings.
        """
        super().__init__(
            metadata=IntegrationMetadata(
                name="web_search",
                display_name="Web Search",
                description=(
                    "Search the web, retrieve recent news stories, and extract clean "
                    "readable text or markdown from public webpages."
                ),
                version="0.1.0",
                auth_required=True,
                auth_type="api_key",
                documentation_url="https://docs.tavily.com",
            )
        )
        self._settings = settings or get_settings()
        self.client = client or WebSearchClient(settings=self._settings)

    def is_configured(self) -> bool:
        """Check whether Tavily API credentials are configured."""
        if getattr(self.client, "is_configured", False):
            return True
        return self._settings.tavily_api_key is not None

    async def test_connection(self) -> bool:
        """Verify that web search service is accessible."""
        if not self.is_configured():
            return False
        return await self.client.probe()

    async def health_check(self) -> IntegrationStatus:
        """Return operational health status for Web Search."""
        if not self.is_configured():
            return IntegrationStatus(
                name=self.name,
                display_name=self.display_name,
                state=IntegrationState.UNCONFIGURED,
                is_configured=False,
                is_connected=False,
                registered_tools=[],
                error_message=(
                    "Tavily API key is not configured. Set SYNQRA_TAVILY_API_KEY in environment."
                ),
                last_checked_at=utc_now(),
            )

        try:
            is_connected = await self.test_connection()
            if is_connected:
                return IntegrationStatus(
                    name=self.name,
                    display_name=self.display_name,
                    state=IntegrationState.CONNECTED,
                    is_configured=True,
                    is_connected=True,
                    registered_tools=self.registered_tools,
                    last_checked_at=utc_now(),
                )

            return IntegrationStatus(
                name=self.name,
                display_name=self.display_name,
                state=IntegrationState.ERROR,
                is_configured=True,
                is_connected=False,
                registered_tools=self.registered_tools,
                error_message="Web search connection probe failed.",
                last_checked_at=utc_now(),
            )
        except Exception as exc:
            return IntegrationStatus(
                name=self.name,
                display_name=self.display_name,
                state=IntegrationState.ERROR,
                is_configured=True,
                is_connected=False,
                registered_tools=self.registered_tools,
                error_message=f"Health probe error: {exc}",
                last_checked_at=utc_now(),
            )

    def _register_tools_impl(self, server: FastMCP) -> list[str]:
        """Register the 3 curated Web Search tools on the FastMCP server."""
        client = self.client

        @server.tool(
            name="search_web",
            description=(
                "Search the web for technical documentation, tutorials, articles, and guides. "
                "Supports configurable result limits, search depth, and domain filtering."
            ),
        )
        async def search_web_tool(
            query: str,
            max_results: int = 5,
            search_depth: str = "basic",
            include_domains: list[str] | None = None,
            exclude_domains: list[str] | None = None,
        ) -> SearchWebOutput:
            """Search the web using Tavily."""
            return await search_web(
                client=client,
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
            )

        @server.tool(
            name="extract_webpage_content",
            description=(
                "Retrieve and extract clean, readable text or markdown content from a public "
                "webpage URL. Guards against SSRF and internal network destinations."
            ),
        )
        async def extract_webpage_content_tool(
            url: str,
        ) -> ExtractWebpageOutput:
            """Extract content from a public webpage."""
            return await extract_webpage_content(
                client=client,
                url=url,
            )

        @server.tool(
            name="search_news",
            description=(
                "Search recent news articles by topic or query, filtered by recency window "
                "(days) and domain restrictions. Returns headlines and summaries."
            ),
        )
        async def search_news_tool(
            query: str,
            max_results: int = 5,
            days: int = 7,
            include_domains: list[str] | None = None,
            exclude_domains: list[str] | None = None,
        ) -> SearchNewsOutput:
            """Search recent news articles."""
            return await search_news(
                client=client,
                query=query,
                max_results=max_results,
                days=days,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
            )

        return [
            "search_web",
            "extract_webpage_content",
            "search_news",
        ]
