"""Web Search tool implementations for SYNQRA.

Provides core business logic for web search, recent news search, and readable webpage
content extraction using Tavily API and safe SSRF validation.
"""

import logging
import urllib.parse

from synqra.integrations.exceptions import IntegrationExecutionError
from synqra.integrations.web_search.client import WebSearchClient
from synqra.integrations.web_search.schemas import (
    ExtractWebpageOutput,
    NewsArticleItem,
    SearchNewsOutput,
    SearchWebOutput,
    WebSearchResultItem,
)

logger = logging.getLogger("synqra.integrations.web_search.tools")

MAX_EXTRACT_CONTENT_LENGTH = 50_000


def extract_domain_from_url(url: str) -> str:
    """Extract a human-readable domain name from a URL.

    Args:
        url: Full webpage URL.

    Returns:
        Domain name string (e.g. 'bbc.com', 'github.com').
    """
    try:
        parsed = urllib.parse.urlparse(url)
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            return netloc[4:]
        return netloc or url
    except Exception:
        return url


async def search_web(
    client: WebSearchClient,
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
) -> SearchWebOutput:
    """Execute a general web search for technical documentation, articles, and guides.

    Args:
        client: Active WebSearchClient instance.
        query: Search keywords or query string.
        max_results: Maximum results to retrieve (1-20).
        search_depth: 'basic' or 'advanced'.
        include_domains: Specific domains to restrict search to.
        exclude_domains: Specific domains to exclude.

    Returns:
        SearchWebOutput containing normalized search results.
    """
    clean_query = query.strip()
    if not clean_query:
        raise IntegrationExecutionError(
            "Search query must not be empty.",
            integration_name="web_search",
        )

    data = await client.search(
        query=clean_query,
        max_results=max_results,
        search_depth=search_depth,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        topic="general",
    )

    raw_results = data.get("results") or []
    normalized: list[WebSearchResultItem] = []

    for item in raw_results:
        if not isinstance(item, dict):
            continue
        item_url = item.get("url") or ""
        normalized.append(
            WebSearchResultItem(
                title=item.get("title") or "Untitled Result",
                url=item_url,
                snippet=item.get("content") or "",
                source=extract_domain_from_url(item_url),
                publication_date=item.get("published_date"),
                score=item.get("score"),
            )
        )

    return SearchWebOutput(
        query=clean_query,
        total_results=len(normalized),
        results=normalized,
    )


async def extract_webpage_content(
    client: WebSearchClient,
    url: str,
    max_length: int = MAX_EXTRACT_CONTENT_LENGTH,
) -> ExtractWebpageOutput:
    """Extract clean, readable text or markdown content from a public webpage URL.

    Args:
        client: Active WebSearchClient instance.
        url: Full HTTP or HTTPS webpage URL.
        max_length: Maximum allowed character length for extracted content.

    Returns:
        ExtractWebpageOutput with clean content and metadata.
    """
    clean_url = url.strip()
    if not clean_url:
        raise IntegrationExecutionError(
            "URL must not be empty.",
            integration_name="web_search",
        )

    data = await client.extract(clean_url)

    failed_results = data.get("failed_results") or []
    results = data.get("results") or []

    if not results and failed_results:
        failed_err = failed_results[0].get("error") if isinstance(failed_results[0], dict) else None
        raise IntegrationExecutionError(
            f"Failed to extract content from '{clean_url}': {failed_err or 'Page inaccessible'}",
            integration_name="web_search",
        )

    first_item = results[0] if results and isinstance(results[0], dict) else {}
    raw_content = first_item.get("raw_content") or ""
    page_title = first_item.get("title")

    raw_len = len(raw_content)
    is_truncated = False
    content = raw_content

    if raw_len > max_length:
        content = raw_content[:max_length]
        is_truncated = True

    return ExtractWebpageOutput(
        url=clean_url,
        title=page_title,
        content=content,
        raw_content_length=raw_len,
        is_truncated=is_truncated,
        metadata={"source": extract_domain_from_url(clean_url)},
    )


async def search_news(
    client: WebSearchClient,
    query: str,
    max_results: int = 5,
    days: int = 7,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
) -> SearchNewsOutput:
    """Search recent news articles with recency filters and domain restrictions.

    Args:
        client: Active WebSearchClient instance.
        query: News topic or query string.
        max_results: Maximum news articles to retrieve (1-20).
        days: Recency window in days (1-30).
        include_domains: Specific news domains to include.
        exclude_domains: Specific news domains to exclude.

    Returns:
        SearchNewsOutput containing normalized news articles.
    """
    clean_query = query.strip()
    if not clean_query:
        raise IntegrationExecutionError(
            "News query must not be empty.",
            integration_name="web_search",
        )

    data = await client.search(
        query=clean_query,
        max_results=max_results,
        topic="news",
        days=days,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
    )

    raw_results = data.get("results") or []
    articles: list[NewsArticleItem] = []

    for item in raw_results:
        if not isinstance(item, dict):
            continue
        item_url = item.get("url") or ""
        articles.append(
            NewsArticleItem(
                headline=item.get("title") or "Untitled News Story",
                source=extract_domain_from_url(item_url),
                publication_date=item.get("published_date"),
                url=item_url,
                summary=item.get("content") or "",
                score=item.get("score"),
            )
        )

    return SearchNewsOutput(
        query=clean_query,
        total_results=len(articles),
        days=days,
        articles=articles,
    )
