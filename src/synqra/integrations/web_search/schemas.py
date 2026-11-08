"""Pydantic schemas for Web Search integration in SYNQRA.

Defines validated input parameters and structured response models for web search,
news article search, and webpage content extraction.
"""

from typing import Any

from pydantic import BaseModel, Field


class WebSearchResultItem(BaseModel):
    """Normalized search result item from the web."""

    title: str = Field(description="Page title.")
    url: str = Field(description="URL of the webpage.")
    snippet: str = Field(description="Informative text snippet or summary from the page.")
    source: str | None = Field(default=None, description="Domain name or publisher of the source.")
    publication_date: str | None = Field(
        default=None, description="Publication timestamp or date if available."
    )
    score: float | None = Field(
        default=None, description="Relevance score assigned to the search result."
    )


class SearchWebInput(BaseModel):
    """Input parameters for searching the web."""

    query: str = Field(
        description="Search terms or questions to find on the web.",
        min_length=1,
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of search results to retrieve (1-20).",
    )
    search_depth: str = Field(
        default="basic",
        description="Search depth: 'basic' for fast retrieval, 'advanced' for thorough research.",
    )
    include_domains: list[str] | None = Field(
        default=None,
        description="Optional list of specific domains to include (e.g. ['python.org']).",
    )
    exclude_domains: list[str] | None = Field(
        default=None,
        description="Optional list of specific domains to exclude.",
    )


class SearchWebOutput(BaseModel):
    """Structured response containing web search results."""

    query: str = Field(description="Original search query.")
    total_results: int = Field(description="Total matching results retrieved.")
    results: list[WebSearchResultItem] = Field(
        default_factory=list, description="List of search result items."
    )


class ExtractWebpageInput(BaseModel):
    """Input parameters for extracting content from a webpage."""

    url: str = Field(
        description="Full HTTP or HTTPS URL of the webpage to extract content from.",
        min_length=1,
    )


class ExtractWebpageOutput(BaseModel):
    """Structured response containing readable extracted content from a webpage."""

    url: str = Field(description="URL of the extracted webpage.")
    title: str | None = Field(default=None, description="Title of the webpage if determined.")
    content: str = Field(description="Clean, readable extracted text or markdown content.")
    raw_content_length: int = Field(description="Total character length of the extracted content.")
    is_truncated: bool = Field(
        default=False,
        description="Whether the content was truncated due to response size limits.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional page metadata.")


class NewsArticleItem(BaseModel):
    """Normalized recent news article item."""

    headline: str = Field(description="Headline of the news article.")
    source: str = Field(description="News publisher or domain name.")
    publication_date: str | None = Field(
        default=None, description="Publication date of the news article."
    )
    url: str = Field(description="Direct URL to the news article.")
    summary: str = Field(description="Summary or content snippet of the article.")
    score: float | None = Field(
        default=None, description="Relevance score assigned to the article."
    )


class SearchNewsInput(BaseModel):
    """Input parameters for searching recent news."""

    query: str = Field(
        description="Keywords or topics for searching recent news stories.",
        min_length=1,
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum news articles to retrieve (1-20).",
    )
    days: int = Field(
        default=7,
        ge=1,
        le=30,
        description="Recency window in days (e.g. 1 for last 24h, 7 for last week).",
    )
    include_domains: list[str] | None = Field(
        default=None,
        description="Optional list of specific news domains to include.",
    )
    exclude_domains: list[str] | None = Field(
        default=None,
        description="Optional list of specific news domains to exclude.",
    )


class SearchNewsOutput(BaseModel):
    """Structured response containing recent news articles."""

    query: str = Field(description="News query searched.")
    total_results: int = Field(description="Total news articles retrieved.")
    days: int = Field(description="Recency window in days queried.")
    articles: list[NewsArticleItem] = Field(
        default_factory=list, description="List of recent news articles."
    )
