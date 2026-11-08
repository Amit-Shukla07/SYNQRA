"""Web Search service integration module for SYNQRA."""

from synqra.integrations.web_search.client import WebSearchClient, validate_safe_url
from synqra.integrations.web_search.integration import WebSearchIntegration
from synqra.integrations.web_search.schemas import (
    ExtractWebpageInput,
    ExtractWebpageOutput,
    NewsArticleItem,
    SearchNewsInput,
    SearchNewsOutput,
    SearchWebInput,
    SearchWebOutput,
    WebSearchResultItem,
)
from synqra.integrations.web_search.tools import (
    extract_webpage_content,
    search_news,
    search_web,
)

__all__ = [
    "ExtractWebpageInput",
    "ExtractWebpageOutput",
    "NewsArticleItem",
    "SearchNewsInput",
    "SearchNewsOutput",
    "SearchWebInput",
    "SearchWebOutput",
    "WebSearchResultItem",
    "WebSearchClient",
    "WebSearchIntegration",
    "extract_webpage_content",
    "search_news",
    "search_web",
    "validate_safe_url",
]
