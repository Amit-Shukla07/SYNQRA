"""Asynchronous HTTP client for Web Search integration using Tavily API.

Provides strongly-typed, SSRF-protected, and rate-limited access to Tavily's
web search, news search, and webpage content extraction endpoints.
"""

import ipaddress
import logging
import socket
import urllib.parse
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx

from synqra.core.config import Settings, get_settings
from synqra.integrations.exceptions import (
    IntegrationConfigurationError,
    IntegrationConnectionError,
    IntegrationExecutionError,
)

logger = logging.getLogger("synqra.integrations.web_search.client")

TAVILY_API_BASE = "https://api.tavily.com"
TAVILY_SEARCH_ENDPOINT = f"{TAVILY_API_BASE}/search"
TAVILY_EXTRACT_ENDPOINT = f"{TAVILY_API_BASE}/extract"

BLOCKED_HOSTS = {
    "localhost",
    "metadata.google.internal",
    "instance-data",
}


def validate_safe_url(url: str) -> str:
    """Validate that a URL is safe and does not target internal or private network destinations.

    Guards against Server-Side Request Forgery (SSRF) and access to loopback,
    private, link-local, and cloud metadata IP ranges.

    Args:
        url: The candidate URL to validate.

    Returns:
        The validated clean URL string.

    Raises:
        IntegrationExecutionError: If the URL is malformed, uses an unsupported scheme,
            or resolves to a private/local network address.
    """
    clean_url = url.strip()
    if not clean_url:
        raise IntegrationExecutionError(
            "URL must not be empty.",
            integration_name="web_search",
        )

    try:
        parsed = urllib.parse.urlparse(clean_url)
    except Exception as exc:
        raise IntegrationExecutionError(
            f"Malformed URL '{clean_url}': {exc}",
            integration_name="web_search",
        ) from exc

    if parsed.scheme.lower() not in ("http", "https"):
        raise IntegrationExecutionError(
            f"Unsupported URL scheme '{parsed.scheme}'. Only 'http' and 'https' are allowed.",
            integration_name="web_search",
        )

    host = parsed.hostname
    if not host:
        raise IntegrationExecutionError(
            f"Missing hostname in URL '{clean_url}'.",
            integration_name="web_search",
        )

    lower_host = host.lower()
    if (
        lower_host in BLOCKED_HOSTS
        or lower_host.endswith(".local")
        or lower_host.endswith(".internal")
        or lower_host.endswith(".localhost")
    ):
        raise IntegrationExecutionError(
            f"Access to private/local network host '{host}' is forbidden.",
            integration_name="web_search",
        )

    # Check direct IP address literals
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local:
            raise IntegrationExecutionError(
                f"Access to private/loopback IP address '{host}' is forbidden.",
                integration_name="web_search",
            )
        return clean_url
    except ValueError:
        pass

    # Check DNS resolution for rebinding / internal hosts
    try:
        addr_info = socket.getaddrinfo(host, None)
        for item in addr_info:
            sockaddr = item[4]
            ip_str = sockaddr[0]
            ip = ipaddress.ip_address(ip_str)
            if ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local:
                raise IntegrationExecutionError(
                    f"Host '{host}' resolves to forbidden private/loopback IP '{ip_str}'.",
                    integration_name="web_search",
                )
    except IntegrationExecutionError:
        raise
    except Exception as exc:
        logger.debug("DNS check error for host '%s': %s", host, exc)

    return clean_url


class WebSearchClient:
    """Async HTTP client for interacting with the Tavily Web Search API."""

    def __init__(
        self,
        api_key: str | None = None,
        settings: Settings | None = None,
        timeout: float = 12.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize the Web Search client.

        Args:
            api_key: Optional Tavily API key. If omitted, loaded from settings.
            settings: Optional application settings.
            timeout: HTTP request timeout in seconds.
            http_client: Optional pre-configured httpx.AsyncClient (for testing/mocking).
        """
        self._settings = settings or get_settings()
        resolved_key = api_key
        if resolved_key is None and self._settings.tavily_api_key is not None:
            resolved_key = self._settings.tavily_api_key.get_secret_value()

        self._api_key = resolved_key
        self.timeout = timeout
        self._http_client = http_client

    @property
    def is_configured(self) -> bool:
        """Return True if a Tavily API key is configured."""
        return bool(self._api_key)

    @asynccontextmanager
    async def _get_client(self) -> AsyncIterator[httpx.AsyncClient]:
        """Provide an active httpx.AsyncClient context."""
        if self._http_client is not None:
            yield self._http_client
        else:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                yield client

    async def probe(self) -> bool:
        """Probe upstream availability and credential readiness."""
        return self.is_configured

    async def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        topic: str = "general",
        days: int | None = None,
    ) -> dict[str, Any]:
        """Execute a web or news search query through Tavily API.

        Args:
            query: Search query string.
            max_results: Maximum results to retrieve (1-20).
            search_depth: 'basic' or 'advanced'.
            include_domains: Optional list of domains to include.
            exclude_domains: Optional list of domains to exclude.
            topic: 'general' for web search, 'news' for recent news.
            days: Recency window in days for news queries.

        Returns:
            Dictionary containing raw response data from Tavily.

        Raises:
            IntegrationConfigurationError: If API key is missing or invalid.
            IntegrationExecutionError: If rate limited or request fails.
            IntegrationConnectionError: If network connection fails.
        """
        if not self.is_configured:
            raise IntegrationConfigurationError(
                "Tavily API key is not configured. Set SYNQRA_TAVILY_API_KEY in environment.",
                integration_name="web_search",
            )

        payload: dict[str, Any] = {
            "api_key": self._api_key,
            "query": query.strip(),
            "search_depth": search_depth,
            "max_results": max(1, min(max_results, 20)),
            "topic": topic,
        }

        if include_domains:
            payload["include_domains"] = [d.strip() for d in include_domains if d.strip()]
        if exclude_domains:
            payload["exclude_domains"] = [d.strip() for d in exclude_domains if d.strip()]
        if days is not None:
            payload["days"] = days

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "SYNQRA/0.1.0",
        }

        try:
            async with self._get_client() as client:
                resp = await client.post(
                    TAVILY_SEARCH_ENDPOINT,
                    json=payload,
                    headers=headers,
                )

            if resp.status_code == 401 or resp.status_code == 403:
                raise IntegrationConfigurationError(
                    f"Tavily API authentication failed ({resp.status_code}): {resp.text}",
                    integration_name="web_search",
                )
            if resp.status_code == 429:
                raise IntegrationExecutionError(
                    "Tavily API rate limit exceeded.",
                    integration_name="web_search",
                )
            if resp.status_code >= 400:
                raise IntegrationExecutionError(
                    f"Tavily search request failed ({resp.status_code}): {resp.text}",
                    integration_name="web_search",
                )

            return resp.json()
        except (IntegrationConfigurationError, IntegrationExecutionError):
            raise
        except httpx.TimeoutException as exc:
            raise IntegrationExecutionError(
                f"Tavily search request timed out after {self.timeout}s: {exc}",
                integration_name="web_search",
            ) from exc
        except Exception as exc:
            raise IntegrationConnectionError(
                f"Failed to connect to Tavily search service: {exc}",
                integration_name="web_search",
            ) from exc

    async def extract(self, url: str) -> dict[str, Any]:
        """Extract readable content from a webpage URL using Tavily Extract API.

        Args:
            url: Validated safe webpage URL.

        Returns:
            Dictionary containing extracted content data.

        Raises:
            IntegrationConfigurationError: If API key is missing or invalid.
            IntegrationExecutionError: If URL is unsafe, extraction fails, or page is inaccessible.
        """
        valid_url = validate_safe_url(url)

        if not self.is_configured:
            raise IntegrationConfigurationError(
                "Tavily API key is not configured. Set SYNQRA_TAVILY_API_KEY in environment.",
                integration_name="web_search",
            )

        payload = {
            "api_key": self._api_key,
            "urls": [valid_url],
        }

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "SYNQRA/0.1.0",
        }

        try:
            async with self._get_client() as client:
                resp = await client.post(
                    TAVILY_EXTRACT_ENDPOINT,
                    json=payload,
                    headers=headers,
                )

            if resp.status_code in (401, 403):
                raise IntegrationConfigurationError(
                    f"Tavily API authentication failed ({resp.status_code}): {resp.text}",
                    integration_name="web_search",
                )
            if resp.status_code == 429:
                raise IntegrationExecutionError(
                    "Tavily API rate limit exceeded.",
                    integration_name="web_search",
                )
            if resp.status_code >= 400:
                raise IntegrationExecutionError(
                    f"Tavily extract request failed ({resp.status_code}): {resp.text}",
                    integration_name="web_search",
                )

            return resp.json()
        except (IntegrationConfigurationError, IntegrationExecutionError):
            raise
        except httpx.TimeoutException as exc:
            raise IntegrationExecutionError(
                f"Webpage extraction request timed out after {self.timeout}s: {exc}",
                integration_name="web_search",
            ) from exc
        except Exception as exc:
            raise IntegrationConnectionError(
                f"Failed to connect to extraction service: {exc}",
                integration_name="web_search",
            ) from exc
