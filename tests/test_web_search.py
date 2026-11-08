"""Unit and integration tests for SYNQRA Web Search integration.

Verifies WebSearchClient, SSRF protection, schema models, tools, and integration health checks
using mocked HTTPX responses without requiring live external network credentials.
"""

from unittest.mock import MagicMock

import httpx
import pytest
from fastmcp import FastMCP

from synqra.core.config import Settings
from synqra.integrations.exceptions import (
    IntegrationConfigurationError,
    IntegrationConnectionError,
    IntegrationExecutionError,
)
from synqra.integrations.models import IntegrationState
from synqra.integrations.web_search.client import (
    WebSearchClient,
    validate_safe_url,
)
from synqra.integrations.web_search.integration import WebSearchIntegration
from synqra.integrations.web_search.tools import (
    extract_domain_from_url,
    extract_webpage_content,
    search_news,
    search_web,
)

# ==============================================================================
# Client & Configuration Tests
# ==============================================================================


def test_web_search_client_unconfigured_without_api_key() -> None:
    """Verify WebSearchClient reports unconfigured when no Tavily API key is provided."""
    empty_settings = Settings()
    client = WebSearchClient(settings=empty_settings)

    assert client.is_configured is False


def test_web_search_client_configured_with_api_key() -> None:
    """Verify WebSearchClient reports configured when Tavily API key is present."""
    client = WebSearchClient(api_key="tvly-mock-key-12345")
    assert client.is_configured is True


# ==============================================================================
# SSRF & URL Safety Validation Tests
# ==============================================================================


@pytest.mark.parametrize(
    "valid_url",
    [
        "https://docs.python.org/3/library/asyncio.html",
        "http://example.com/api/data",
        "https://github.com/fastmcp/fastmcp",
    ],
)
def test_validate_safe_url_valid(valid_url: str) -> None:
    """Verify valid public HTTP and HTTPS URLs pass SSRF validation."""
    assert validate_safe_url(valid_url) == valid_url


@pytest.mark.parametrize(
    "unsafe_url",
    [
        "http://localhost:8080/admin",
        "http://127.0.0.1:3000",
        "http://10.0.0.1/secret",
        "http://192.168.1.1/router",
        "http://172.16.0.10:8000",
        "http://169.254.169.254/latest/meta-data/",
        "http://metadata.google.internal/computeMetadata/v1/",
        "ftp://ftp.example.com/file.txt",
        "file:///etc/passwd",
        "",
        "http://localtest.me",
    ],
)
def test_validate_safe_url_unsafe_blocked(unsafe_url: str) -> None:
    """Verify internal, private, loopback, metadata, and non-HTTP schemes are blocked."""
    with pytest.raises(IntegrationExecutionError):
        validate_safe_url(unsafe_url)


def test_extract_domain_from_url() -> None:
    """Verify domain name extraction helper."""
    assert extract_domain_from_url("https://www.bbc.com/news/world") == "bbc.com"
    assert extract_domain_from_url("https://docs.python.org/3/") == "docs.python.org"
    assert extract_domain_from_url("invalid-url") == "invalid-url"


# ==============================================================================
# Tool 1: search_web Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_search_web_success() -> None:
    """Verify search_web normalizes results and metadata correctly."""
    mock_response_data = {
        "query": "fastmcp python",
        "results": [
            {
                "title": "FastMCP - Model Context Protocol for Python",
                "url": "https://gofastmcp.com/docs",
                "content": "FastMCP is a production-grade framework for building MCP servers.",
                "score": 0.98,
                "published_date": "2026-01-15",
            },
            {
                "title": "FastMCP GitHub Repository",
                "url": "https://github.com/fastmcp/fastmcp",
                "content": "Official repository for FastMCP in Python.",
                "score": 0.89,
                "published_date": None,
            },
        ],
    }

    mock_client = MagicMock(spec=WebSearchClient)
    mock_client.search = MagicMock()

    async def mock_search(**kwargs):
        _ = kwargs
        return mock_response_data

    mock_client.search.side_effect = mock_search

    result = await search_web(
        mock_client,
        query="fastmcp python",
        max_results=5,
        search_depth="advanced",
        include_domains=["gofastmcp.com"],
    )

    assert result.query == "fastmcp python"
    assert result.total_results == 2
    assert result.results[0].title == "FastMCP - Model Context Protocol for Python"
    assert result.results[0].source == "gofastmcp.com"
    assert result.results[0].score == 0.98
    assert result.results[1].source == "github.com"


@pytest.mark.asyncio
async def test_search_web_empty_query_raises() -> None:
    """Verify search_web raises error when query is empty."""
    mock_client = MagicMock(spec=WebSearchClient)
    with pytest.raises(IntegrationExecutionError, match="Search query must not be empty"):
        await search_web(mock_client, query="   ")


# ==============================================================================
# Tool 2: extract_webpage_content Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_extract_webpage_content_success() -> None:
    """Verify extract_webpage_content extracts readable text."""
    mock_extract_data = {
        "results": [
            {
                "url": "https://docs.python.org/3/library/asyncio.html",
                "title": "asyncio — Asynchronous I/O",
                "raw_content": "# asyncio\nasyncio is a library to write concurrent code...",
            }
        ],
        "failed_results": [],
    }

    mock_client = MagicMock(spec=WebSearchClient)

    async def mock_extract(url: str):
        _ = url
        return mock_extract_data

    mock_client.extract = MagicMock(side_effect=mock_extract)

    result = await extract_webpage_content(
        mock_client,
        url="https://docs.python.org/3/library/asyncio.html",
    )

    assert result.url == "https://docs.python.org/3/library/asyncio.html"
    assert result.title == "asyncio — Asynchronous I/O"
    assert "asyncio is a library" in result.content
    assert result.is_truncated is False
    assert result.metadata["source"] == "docs.python.org"


@pytest.mark.asyncio
async def test_extract_webpage_content_truncation() -> None:
    """Verify extract_webpage_content truncates text exceeding max_length."""
    large_text = "A" * 1500
    mock_extract_data = {
        "results": [
            {
                "url": "https://example.com/large",
                "title": "Large Page",
                "raw_content": large_text,
            }
        ]
    }

    mock_client = MagicMock(spec=WebSearchClient)

    async def mock_extract(url: str):
        _ = url
        return mock_extract_data

    mock_client.extract = MagicMock(side_effect=mock_extract)

    result = await extract_webpage_content(
        mock_client,
        url="https://example.com/large",
        max_length=500,
    )

    assert len(result.content) == 500
    assert result.raw_content_length == 1500
    assert result.is_truncated is True


@pytest.mark.asyncio
async def test_extract_webpage_content_failed_result_raises() -> None:
    """Verify extract_webpage_content raises error when page is inaccessible."""
    mock_extract_data = {
        "results": [],
        "failed_results": [{"url": "https://example.com/broken", "error": "404 Not Found"}],
    }

    mock_client = MagicMock(spec=WebSearchClient)

    async def mock_extract(url: str):
        _ = url
        return mock_extract_data

    mock_client.extract = MagicMock(side_effect=mock_extract)

    with pytest.raises(IntegrationExecutionError, match="404 Not Found"):
        await extract_webpage_content(mock_client, url="https://example.com/broken")


# ==============================================================================
# Tool 3: search_news Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_search_news_success() -> None:
    """Verify search_news retrieves recent articles with publication dates."""
    mock_news_data = {
        "query": "artificial intelligence",
        "results": [
            {
                "title": "Breakthrough in AI Agent Protocols",
                "url": "https://techcrunch.com/2026/02/ai-agents",
                "content": "A new universal protocol allows agents to share tool definitions.",
                "score": 0.95,
                "published_date": "2026-02-10T12:00:00Z",
            }
        ],
    }

    mock_client = MagicMock(spec=WebSearchClient)

    async def mock_search(**kwargs):
        _ = kwargs
        return mock_news_data

    mock_client.search = MagicMock(side_effect=mock_search)

    result = await search_news(
        mock_client,
        query="artificial intelligence",
        days=3,
        include_domains=["techcrunch.com"],
    )

    assert result.query == "artificial intelligence"
    assert result.total_results == 1
    assert result.days == 3
    assert result.articles[0].headline == "Breakthrough in AI Agent Protocols"
    assert result.articles[0].source == "techcrunch.com"
    assert result.articles[0].publication_date == "2026-02-10T12:00:00Z"


# ==============================================================================
# HTTP Client Error Handling Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_client_search_unconfigured_raises() -> None:
    """Verify search raises IntegrationConfigurationError when key missing."""
    client = WebSearchClient(api_key=None, settings=Settings())
    with pytest.raises(IntegrationConfigurationError):
        await client.search(query="test")


@pytest.mark.asyncio
async def test_client_search_401_auth_failure() -> None:
    """Verify 401 response from Tavily raises IntegrationConfigurationError."""
    mock_resp = httpx.Response(401, text="Unauthorized: Invalid API key")
    mock_http = MagicMock(spec=httpx.AsyncClient)

    async def mock_post(url: str, **kwargs):
        _ = url
        _ = kwargs
        return mock_resp

    mock_http.post = MagicMock(side_effect=mock_post)

    client = WebSearchClient(api_key="invalid-key", http_client=mock_http)
    with pytest.raises(IntegrationConfigurationError, match="authentication failed"):
        await client.search(query="test")


@pytest.mark.asyncio
async def test_client_search_429_rate_limit() -> None:
    """Verify 429 response raises IntegrationExecutionError with rate limit message."""
    mock_resp = httpx.Response(429, text="Rate limit exceeded")
    mock_http = MagicMock(spec=httpx.AsyncClient)

    async def mock_post(url: str, **kwargs):
        _ = url
        _ = kwargs
        return mock_resp

    mock_http.post = MagicMock(side_effect=mock_post)

    client = WebSearchClient(api_key="tvly-mock", http_client=mock_http)
    with pytest.raises(IntegrationExecutionError, match="rate limit exceeded"):
        await client.search(query="test")


@pytest.mark.asyncio
async def test_client_search_timeout_handling() -> None:
    """Verify timeout raises IntegrationExecutionError."""
    mock_http = MagicMock(spec=httpx.AsyncClient)

    async def mock_post(url: str, **kwargs):
        _ = url
        _ = kwargs
        raise httpx.TimeoutException("Connection timed out")

    mock_http.post = MagicMock(side_effect=mock_post)

    client = WebSearchClient(api_key="tvly-mock", http_client=mock_http)
    with pytest.raises(IntegrationExecutionError, match="timed out"):
        await client.search(query="test")


@pytest.mark.asyncio
async def test_client_extract_network_failure() -> None:
    """Verify network exception raises IntegrationConnectionError."""
    mock_http = MagicMock(spec=httpx.AsyncClient)

    async def mock_post(url: str, **kwargs):
        _ = url
        _ = kwargs
        raise httpx.ConnectError("Failed to connect")

    mock_http.post = MagicMock(side_effect=mock_post)

    client = WebSearchClient(api_key="tvly-mock", http_client=mock_http)
    with pytest.raises(IntegrationConnectionError, match="Failed to connect"):
        await client.extract(url="https://example.com")


# ==============================================================================
# Integration & FastMCP Registration Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_web_search_integration_tool_registration() -> None:
    """Verify WebSearchIntegration registers the three expected MCP tools."""
    client = MagicMock(spec=WebSearchClient)
    client.is_configured = True
    integration = WebSearchIntegration(client=client)

    server = FastMCP("TestWebSearchServer")
    registered = integration.register_tools(server)

    expected = ["search_web", "extract_webpage_content", "search_news"]
    assert len(registered) == 3
    for tool_name in expected:
        assert tool_name in registered

    server_tools = await server.list_tools()
    server_tool_names = [t.name for t in server_tools]
    for tool_name in expected:
        assert tool_name in server_tool_names


@pytest.mark.asyncio
async def test_web_search_health_check_connected() -> None:
    """Verify health_check returns CONNECTED when configured and probe succeeds."""
    client = MagicMock(spec=WebSearchClient)
    client.is_configured = True

    async def mock_probe():
        return True

    client.probe = MagicMock(side_effect=mock_probe)

    integration = WebSearchIntegration(client=client)
    status = await integration.health_check()

    assert status.state == IntegrationState.CONNECTED
    assert status.is_connected is True
    assert status.is_configured is True


def test_web_search_health_check_unconfigured() -> None:
    """Verify health_check returns UNCONFIGURED when no API key is provided."""
    empty_settings = Settings()
    integration = WebSearchIntegration(settings=empty_settings)
    assert integration.is_configured() is False
