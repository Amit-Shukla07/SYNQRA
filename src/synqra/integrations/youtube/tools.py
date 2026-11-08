"""Tool implementations for the YouTube integration.

Provides sanitized, well-structured MCP tools for searching YouTube educational
videos and retrieving detailed video metadata via Composio.
"""

import logging
import re
from typing import Any

import httpx

from synqra.integrations.exceptions import IntegrationExecutionError
from synqra.integrations.youtube.client import (
    YOUTUBE_DETAILS_ACTION,
    YOUTUBE_SEARCH_ACTION,
    YouTubeClient,
)
from synqra.integrations.youtube.schemas import (
    YouTubeSearchOutput,
    YouTubeVideoDetails,
    YouTubeVideoItem,
)

logger = logging.getLogger("synqra.integrations.youtube.tools")


def format_iso8601_duration(raw_duration: str | None) -> str | None:
    """Format an ISO 8601 duration ('PT1H23M45S') into a readable string ('1h 23m 45s')."""
    if not raw_duration or not raw_duration.startswith("PT"):
        return raw_duration

    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", raw_duration)
    if not match:
        return raw_duration

    hours, minutes, seconds = match.groups()
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if seconds:
        parts.append(f"{seconds}s")

    return " ".join(parts) if parts else "0s"


def _extract_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract item records from various Composio response envelope structures."""
    if not isinstance(payload, dict):
        return []

    # Check direct items
    if "items" in payload and isinstance(payload["items"], list):
        return payload["items"]

    # Check nested data.items
    data = payload.get("data")
    if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
        return data["items"]

    # Check nested response_data.items
    resp_data = payload.get("response_data")
    if isinstance(resp_data, dict):
        items = resp_data.get("items")
        if isinstance(items, list):
            return items

    # Check results list
    results = payload.get("results")
    if isinstance(results, list):
        return results

    return []


async def search_youtube_videos(
    client: YouTubeClient,
    query: str,
    max_results: int = 5,
    order: str = "relevance",
) -> YouTubeSearchOutput:
    """Search YouTube for educational videos, tutorials, and lecture content.

    Args:
        client: Configured YouTubeClient instance.
        query: Search keywords or query string.
        max_results: Maximum videos to return (1-50).
        order: Sort order ('relevance', 'date', 'viewCount', 'rating').

    Returns:
        YouTubeSearchOutput containing structured video items.
    """
    clean_query = query.strip()
    if not clean_query:
        return YouTubeSearchOutput(query=query, total_results=0, videos=[])

    limit = max(1, min(max_results, 50))
    params = {
        "q": clean_query,
        "maxResults": limit,
        "order": order,
        "type": "video",
        "part": "snippet",
    }

    raw_response = await client.execute_tool_async(
        tool_slug=YOUTUBE_SEARCH_ACTION,
        arguments=params,
    )

    items = _extract_items(raw_response)
    video_items: list[YouTubeVideoItem] = []

    for item in items:
        # Extract video ID (can be string or dict with videoId)
        raw_id = item.get("id")
        video_id = ""
        if isinstance(raw_id, dict):
            video_id = raw_id.get("videoId") or raw_id.get("id") or ""
        elif isinstance(raw_id, str):
            video_id = raw_id

        if not video_id:
            continue

        snippet = item.get("snippet", {})
        content_details = item.get("contentDetails", {})
        raw_duration = content_details.get("duration")

        video_items.append(
            YouTubeVideoItem(
                video_id=video_id,
                title=snippet.get("title", "Untitled Video"),
                url=f"https://www.youtube.com/watch?v={video_id}",
                channel_name=snippet.get("channelTitle", "Unknown Channel"),
                description=snippet.get("description", ""),
                published_at=snippet.get("publishedAt"),
                duration=format_iso8601_duration(raw_duration),
            )
        )

    return YouTubeSearchOutput(
        query=clean_query,
        total_results=len(video_items),
        videos=video_items[:limit],
    )


async def _fetch_oembed_details(video_id: str) -> dict[str, Any]:
    """Fetch public unmetered video metadata from YouTube oEmbed API as a resilient fallback."""
    url = (
        f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
    )
    try:
        async with httpx.AsyncClient(timeout=6.0) as http_client:
            resp = await http_client.get(url)
            if resp.status_code == 200:
                return resp.json()
    except Exception as exc:
        logger.debug("oEmbed fetch failed for %s: %s", video_id, exc)
    return {}


async def get_video_details(
    client: YouTubeClient,
    video_id: str,
) -> YouTubeVideoDetails:
    """Retrieve detailed metadata for a single YouTube video.

    Args:
        client: Configured YouTubeClient instance.
        video_id: The unique 11-character YouTube video ID.

    Returns:
        YouTubeVideoDetails model with metadata, metrics, and description.

    Raises:
        IntegrationExecutionError: If video is not found or request fails.
    """
    clean_id = video_id.strip()
    if not clean_id:
        raise IntegrationExecutionError(
            "Video ID must not be empty.",
            integration_name="youtube",
        )

    params = {
        "id": [clean_id],
        "parts": ["snippet", "statistics"],
    }

    try:
        raw_response = await client.execute_tool_async(
            tool_slug=YOUTUBE_DETAILS_ACTION,
            arguments=params,
        )
    except IntegrationExecutionError as exc:
        if "quotaExceeded" in str(exc) or "quota" in str(exc).lower():
            logger.info("Composio YouTube quota exceeded; attempting snippet or oEmbed fallback.")
            try:
                raw_response = await client.execute_tool_async(
                    tool_slug=YOUTUBE_DETAILS_ACTION,
                    arguments={"id": [clean_id], "parts": ["snippet"]},
                )
            except Exception:
                oembed = await _fetch_oembed_details(clean_id)
                if oembed:
                    author = oembed.get("author_name", "Unknown Channel")
                    return YouTubeVideoDetails(
                        video_id=clean_id,
                        title=oembed.get("title", "Untitled Video"),
                        url=f"https://www.youtube.com/watch?v={clean_id}",
                        description=f"Video by {author} on YouTube.",
                        channel_name=author,
                        published_at=None,
                        duration=None,
                        view_count=None,
                        like_count=None,
                    )
                raise
        else:
            raise

    items = _extract_items(raw_response)
    if not items:
        oembed = await _fetch_oembed_details(clean_id)
        if oembed:
            author = oembed.get("author_name", "Unknown Channel")
            return YouTubeVideoDetails(
                video_id=clean_id,
                title=oembed.get("title", "Untitled Video"),
                url=f"https://www.youtube.com/watch?v={clean_id}",
                description=f"Video by {author} on YouTube.",
                channel_name=author,
                published_at=None,
                duration=None,
                view_count=None,
                like_count=None,
            )
        raise IntegrationExecutionError(
            f"No video found matching ID '{clean_id}'.",
            integration_name="youtube",
        )

    item = items[0]
    snippet = item.get("snippet", {})
    content_details = item.get("contentDetails", {})
    statistics = item.get("statistics", {})

    raw_views = statistics.get("viewCount")
    view_count = int(raw_views) if raw_views and str(raw_views).isdigit() else None

    raw_likes = statistics.get("likeCount")
    like_count = int(raw_likes) if raw_likes and str(raw_likes).isdigit() else None

    raw_duration = content_details.get("duration")

    return YouTubeVideoDetails(
        video_id=clean_id,
        title=snippet.get("title", "Untitled Video"),
        url=f"https://www.youtube.com/watch?v={clean_id}",
        description=snippet.get("description", ""),
        channel_name=snippet.get("channelTitle", "Unknown Channel"),
        published_at=snippet.get("publishedAt"),
        duration=format_iso8601_duration(raw_duration),
        view_count=view_count,
        like_count=like_count,
    )
