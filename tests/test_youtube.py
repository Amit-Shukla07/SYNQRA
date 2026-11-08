"""Unit and integration tests for YouTube integration using Composio's Session API."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastmcp import FastMCP

from synqra.core.config import Settings
from synqra.integrations.exceptions import (
    IntegrationConfigurationError,
    IntegrationExecutionError,
)
from synqra.integrations.models import IntegrationState
from synqra.integrations.youtube.client import (
    YOUTUBE_SEARCH_ACTION,
    YouTubeClient,
)
from synqra.integrations.youtube.integration import YouTubeIntegration
from synqra.integrations.youtube.tools import (
    format_iso8601_duration,
    get_video_details,
    search_youtube_videos,
)

MOCK_SEARCH_PAYLOAD = {
    "items": [
        {
            "id": {"videoId": "vid_1", "kind": "youtube#video"},
            "snippet": {
                "title": "Python for Beginners",
                "description": "Learn Python step by step.",
                "channelTitle": "Python Academy",
                "publishedAt": "2024-01-15T12:00:00Z",
            },
            "contentDetails": {"duration": "PT15M30S"},
        },
        {
            "id": {"videoId": "vid_2", "kind": "youtube#video"},
            "snippet": {
                "title": "Advanced Python Patterns",
                "description": "Deep dive into Python architecture.",
                "channelTitle": "Tech Talks",
                "publishedAt": "2024-02-01T09:00:00Z",
            },
            "contentDetails": {"duration": "PT1H10M"},
        },
    ]
}

MOCK_DETAILS_PAYLOAD = {
    "items": [
        {
            "id": "vid_1",
            "snippet": {
                "title": "Python for Beginners",
                "description": "Comprehensive tutorial description.",
                "channelTitle": "Python Academy",
                "publishedAt": "2024-01-15T12:00:00Z",
            },
            "contentDetails": {"duration": "PT15M30S"},
            "statistics": {
                "viewCount": "250000",
                "likeCount": "12000",
            },
        }
    ]
}


def test_duration_formatting() -> None:
    """Verify ISO 8601 duration formatting helper."""
    assert format_iso8601_duration("PT1H23M45S") == "1h 23m 45s"
    assert format_iso8601_duration("PT15M") == "15m"
    assert format_iso8601_duration("PT45S") == "45s"
    assert format_iso8601_duration("PT2H") == "2h"
    assert format_iso8601_duration(None) is None
    assert format_iso8601_duration("raw_string") == "raw_string"


def test_youtube_client_missing_credentials() -> None:
    """Verify that unconfigured client raises IntegrationConfigurationError."""
    settings = Settings(composio_api_key=None)
    client = YouTubeClient(api_key=None, settings=settings)

    assert client.check_connection() is False
    with pytest.raises(IntegrationConfigurationError) as exc_info:
        client.get_or_create_session()
    assert "Composio API key is not configured" in str(exc_info.value)


def test_youtube_client_session_creation() -> None:
    """Verify that client creates session with stable user_id and youtube toolkit."""
    mock_composio = MagicMock()
    mock_session = MagicMock()
    mock_composio.create.return_value = mock_session

    client = YouTubeClient(
        api_key="mock_api_key",
        user_id="stable_test_user",
        client=mock_composio,
    )

    session = client.get_or_create_session()
    assert session is mock_session
    mock_composio.create.assert_called_once_with(
        user_id="stable_test_user",
        toolkits=["youtube"],
    )


def test_youtube_client_connection_active() -> None:
    """Verify check_connection returns True when toolkit state is active."""
    mock_session = MagicMock()
    mock_item = SimpleNamespace(
        slug="youtube",
        connection=SimpleNamespace(is_active=True),
    )
    mock_session.toolkits.return_value = SimpleNamespace(items=[mock_item])

    client = YouTubeClient(session=mock_session)
    assert client.check_connection() is True
    mock_session.toolkits.assert_called_once_with(toolkits=["youtube"])


def test_youtube_client_connection_inactive() -> None:
    """Verify check_connection returns False when connection is not active."""
    mock_session = MagicMock()
    mock_item = SimpleNamespace(
        slug="youtube",
        connection=SimpleNamespace(is_active=False),
    )
    mock_session.toolkits.return_value = SimpleNamespace(items=[mock_item])

    client = YouTubeClient(session=mock_session)
    assert client.check_connection() is False


def test_youtube_client_authorize() -> None:
    """Verify authorize calls session.authorize('youtube') and captures redirect_url."""
    mock_session = MagicMock()
    mock_request = SimpleNamespace(
        id="req_123",
        redirect_url="https://composio.dev/auth/youtube_connect_test",
    )
    mock_session.authorize.return_value = mock_request

    client = YouTubeClient(session=mock_session)
    url = client.authorize(callback_url="https://app.synqra.ai/callback")

    assert url == "https://composio.dev/auth/youtube_connect_test"
    assert client.authorization_url == "https://composio.dev/auth/youtube_connect_test"
    mock_session.authorize.assert_called_once_with(
        "youtube",
        callback_url="https://app.synqra.ai/callback",
    )


def test_youtube_client_discover_tools() -> None:
    """Verify tool discovery returns slugs from session search."""
    mock_session = MagicMock()
    mock_item1 = SimpleNamespace(slug="YOUTUBE_SEARCH")
    mock_item2 = SimpleNamespace(slug="YOUTUBE_VIDEOS_LIST")
    mock_session.search.return_value = SimpleNamespace(items=[mock_item1, mock_item2])

    client = YouTubeClient(session=mock_session)
    tools = client.discover_tools()
    assert "YOUTUBE_SEARCH" in tools
    assert "YOUTUBE_VIDEOS_LIST" in tools


def test_youtube_client_execute_tool_success() -> None:
    """Verify tool execution returns data payload when completed."""
    mock_session = MagicMock()
    mock_response = SimpleNamespace(
        data=MOCK_SEARCH_PAYLOAD,
        error=None,
        result_type="completed",
    )
    mock_session.execute.return_value = mock_response

    client = YouTubeClient(session=mock_session)
    result = client.execute_tool(YOUTUBE_SEARCH_ACTION, {"q": "python"})
    assert result == MOCK_SEARCH_PAYLOAD
    mock_session.execute.assert_called_once_with(
        tool_slug=YOUTUBE_SEARCH_ACTION,
        arguments={"q": "python"},
    )


def test_youtube_client_execute_tool_failure() -> None:
    """Verify tool execution raises IntegrationExecutionError on failed status."""
    mock_session = MagicMock()
    mock_response = SimpleNamespace(
        data=None,
        error="YouTube quota exceeded",
        result_type="failed",
    )
    mock_session.execute.return_value = mock_response

    client = YouTubeClient(session=mock_session)
    with pytest.raises(IntegrationExecutionError) as exc_info:
        client.execute_tool(YOUTUBE_SEARCH_ACTION, {"q": "test"})
    assert "quota exceeded" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_search_youtube_videos_success() -> None:
    """Verify search_youtube_videos parses session response into YouTubeSearchOutput."""
    mock_client = MagicMock(spec=YouTubeClient)
    mock_client.execute_tool_async.return_value = MOCK_SEARCH_PAYLOAD

    result = await search_youtube_videos(
        client=mock_client,
        query="python",
        max_results=2,
    )

    assert result.query == "python"
    assert result.total_results == 2
    assert len(result.videos) == 2

    v1 = result.videos[0]
    assert v1.video_id == "vid_1"
    assert v1.title == "Python for Beginners"
    assert v1.url == "https://www.youtube.com/watch?v=vid_1"
    assert v1.channel_name == "Python Academy"
    assert v1.duration == "15m 30s"

    v2 = result.videos[1]
    assert v2.video_id == "vid_2"
    assert v2.duration == "1h 10m"


@pytest.mark.asyncio
async def test_search_youtube_videos_empty_query() -> None:
    """Verify empty query returns empty result without calling execute_tool."""
    mock_client = MagicMock(spec=YouTubeClient)
    result = await search_youtube_videos(client=mock_client, query="   ")

    assert result.total_results == 0
    assert result.videos == []
    mock_client.execute_tool_async.assert_not_called()


@pytest.mark.asyncio
async def test_get_video_details_success() -> None:
    """Verify get_video_details parses metadata and statistics into YouTubeVideoDetails."""
    mock_client = MagicMock(spec=YouTubeClient)
    mock_client.execute_tool_async.return_value = MOCK_DETAILS_PAYLOAD

    details = await get_video_details(client=mock_client, video_id="vid_1")

    assert details.video_id == "vid_1"
    assert details.title == "Python for Beginners"
    assert details.url == "https://www.youtube.com/watch?v=vid_1"
    assert details.channel_name == "Python Academy"
    assert details.view_count == 250000
    assert details.like_count == 12000
    assert details.duration == "15m 30s"


@pytest.mark.asyncio
async def test_get_video_details_not_found() -> None:
    """Verify get_video_details raises error when video is not found."""
    mock_client = MagicMock(spec=YouTubeClient)
    mock_client.execute_tool_async.return_value = {"items": []}

    with pytest.raises(IntegrationExecutionError) as exc_info:
        await get_video_details(client=mock_client, video_id="nonexistent")
    assert "No video found matching ID" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_video_details_empty_id() -> None:
    """Verify get_video_details raises error on blank ID."""
    mock_client = MagicMock(spec=YouTubeClient)
    with pytest.raises(IntegrationExecutionError) as exc_info:
        await get_video_details(client=mock_client, video_id="  ")
    assert "Video ID must not be empty" in str(exc_info.value)


@pytest.mark.asyncio
async def test_youtube_integration_health_check_flow() -> None:
    """Verify health_check reporting: unconfigured, unauthenticated with auth URL, connected."""
    # 1. Unconfigured
    unconf_client = MagicMock(spec=YouTubeClient)
    unconf_client.is_configured = False
    unconf_integration = YouTubeIntegration(client=unconf_client)
    status_unconf = await unconf_integration.health_check()
    assert status_unconf.state == IntegrationState.UNCONFIGURED
    assert status_unconf.is_connected is False

    # 2. Configured but not connected (must provide authorization URL)
    auth_client = MagicMock(spec=YouTubeClient)
    auth_client.is_configured = True
    auth_client.check_connection_async.return_value = False
    auth_client.authorize_async.return_value = "https://composio.dev/auth/youtube_link"
    auth_integration = YouTubeIntegration(client=auth_client)
    status_auth = await auth_integration.health_check()
    assert status_auth.state == IntegrationState.CONFIGURED
    assert status_auth.is_connected is False
    assert "https://composio.dev/auth/youtube_link" in str(status_auth.error_message)

    # 3. Connected and verified
    conn_client = MagicMock(spec=YouTubeClient)
    conn_client.is_configured = True
    conn_client.check_connection_async.return_value = True
    conn_integration = YouTubeIntegration(client=conn_client)
    status_conn = await conn_integration.health_check()
    assert status_conn.state == IntegrationState.CONNECTED
    assert status_conn.is_connected is True
    assert status_conn.error_message is None


@pytest.mark.asyncio
async def test_mcp_tool_registration_and_execution() -> None:
    """Verify that FastMCP registers YouTube MCP tools and executes calls."""
    mock_client = MagicMock(spec=YouTubeClient)
    mock_client.is_configured = True
    mock_client.check_connection_async.return_value = True
    mock_client.execute_tool_async.return_value = MOCK_SEARCH_PAYLOAD

    integration = YouTubeIntegration(client=mock_client)
    server = FastMCP("TestYouTubeServer")

    tool_names = integration.register_tools(server)
    assert "search_youtube_videos" in tool_names
    assert "get_video_details" in tool_names

    res = await server.call_tool("search_youtube_videos", {"query": "python"})
    assert res.is_error is False
    assert res.structured_content is not None
    assert res.structured_content["total_results"] == 2
    assert res.structured_content["videos"][0]["video_id"] == "vid_1"
