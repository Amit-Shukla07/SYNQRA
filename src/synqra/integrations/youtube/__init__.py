"""YouTube service integration package for SYNQRA.

Connects AI agents and MCP clients with YouTube via Composio for educational
video discovery and metadata retrieval.
"""

from synqra.integrations.youtube.client import YouTubeClient
from synqra.integrations.youtube.integration import YouTubeIntegration
from synqra.integrations.youtube.schemas import (
    YouTubeSearchInput,
    YouTubeSearchOutput,
    YouTubeVideoDetails,
    YouTubeVideoDetailsInput,
    YouTubeVideoItem,
)
from synqra.integrations.youtube.tools import (
    format_iso8601_duration,
    get_video_details,
    search_youtube_videos,
)

__all__ = [
    "YouTubeClient",
    "YouTubeIntegration",
    "YouTubeSearchInput",
    "YouTubeSearchOutput",
    "YouTubeVideoDetails",
    "YouTubeVideoDetailsInput",
    "YouTubeVideoItem",
    "format_iso8601_duration",
    "get_video_details",
    "search_youtube_videos",
]
