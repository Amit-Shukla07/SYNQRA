"""YouTube service integration connector for SYNQRA via Composio Session API."""

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
from synqra.integrations.youtube.client import YouTubeClient
from synqra.integrations.youtube.schemas import YouTubeSearchOutput, YouTubeVideoDetails
from synqra.integrations.youtube.tools import get_video_details, search_youtube_videos

logger = logging.getLogger("synqra.integrations.youtube.integration")


class YouTubeIntegration(BaseIntegration):
    """Integration connector for YouTube video discovery and metadata extraction via Composio."""

    def __init__(
        self,
        client: YouTubeClient | None = None,
        settings: Settings | None = None,
        user_id: str | None = None,
    ) -> None:
        super().__init__(
            metadata=IntegrationMetadata(
                name="youtube",
                display_name="YouTube",
                description="Search educational videos, tutorials, and lecture transcripts.",
                version="0.1.0",
                auth_required=True,
                auth_type="composio",
                documentation_url="https://composio.dev/tools/youtube",
            )
        )
        self._settings = settings or get_settings()
        self.client = client or YouTubeClient(
            settings=self._settings,
            user_id=user_id or self._settings.composio_user_id,
        )

    def is_configured(self) -> bool:
        """Check whether the required Composio credentials are configured."""
        if getattr(self.client, "is_configured", False):
            return True
        return self._settings.composio_api_key is not None

    async def test_connection(self) -> bool:
        """Verify that a valid, active YouTube connection exists through Composio."""
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
            logger.debug("Failed to obtain YouTube authorization URL: %s", exc)

        err_msg = (
            f"YouTube account not connected. Authorize at: {auth_url}"
            if auth_url
            else "YouTube account not connected. Authorize via Composio."
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
        """Register SYNQRA YouTube MCP tools onto the FastMCP server."""

        @server.tool(
            name="search_youtube_videos",
            description=(
                "Search YouTube for educational videos, lecture recordings, tutorials, "
                "and learning materials. Returns video titles, watch URLs, channel names, "
                "and descriptions."
            ),
        )
        async def search_youtube_videos_tool(
            query: str,
            max_results: int = 5,
            order: str = "relevance",
        ) -> YouTubeSearchOutput:
            """Search YouTube videos by query and return normalized results."""
            return await search_youtube_videos(
                client=self.client,
                query=query,
                max_results=max_results,
                order=order,
            )

        @server.tool(
            name="get_video_details",
            description=(
                "Retrieve detailed metadata for a YouTube video by its video ID, including "
                "title, full description, channel name, publication date, duration, and view count."
            ),
        )
        async def get_video_details_tool(video_id: str) -> YouTubeVideoDetails:
            """Fetch detailed metadata for a YouTube video."""
            return await get_video_details(
                client=self.client,
                video_id=video_id,
            )

        return ["search_youtube_videos", "get_video_details"]
