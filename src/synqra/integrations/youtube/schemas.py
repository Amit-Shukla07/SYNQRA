"""Pydantic schemas for the YouTube integration.

Defines strongly-typed input models and normalized output models for YouTube
video searches and metadata extraction.
"""

from pydantic import BaseModel, Field


class YouTubeVideoItem(BaseModel):
    """Normalized metadata for a YouTube video."""

    video_id: str = Field(description="Unique YouTube video ID.")
    title: str = Field(description="Video title.")
    url: str = Field(description="Direct YouTube watch URL.")
    channel_name: str = Field(description="Channel or creator name.")
    description: str = Field(description="Video description snippet.")
    published_at: str | None = Field(default=None, description="ISO 8601 publication timestamp.")
    duration: str | None = Field(default=None, description="Video duration if available.")


class YouTubeSearchInput(BaseModel):
    """Input parameters for the search_youtube_videos tool."""

    query: str = Field(description="Search keywords, educational topic, or video title.")
    max_results: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of videos to return (1-50).",
    )
    order: str = Field(
        default="relevance",
        description="Sort order: 'relevance', 'date', 'viewCount', or 'rating'.",
    )


class YouTubeSearchOutput(BaseModel):
    """Output model for search_youtube_videos."""

    query: str = Field(description="Original search query executed.")
    total_results: int = Field(description="Count of returned video items.")
    videos: list[YouTubeVideoItem] = Field(
        default_factory=list, description="List of matched YouTube videos."
    )


class YouTubeVideoDetailsInput(BaseModel):
    """Input parameters for the get_video_details tool."""

    video_id: str = Field(description="The unique 11-character YouTube video ID.")


class YouTubeVideoDetails(BaseModel):
    """Detailed metadata for a single YouTube video."""

    video_id: str = Field(description="Unique YouTube video ID.")
    title: str = Field(description="Video title.")
    url: str = Field(description="Direct YouTube watch URL.")
    description: str = Field(description="Full or snippet video description.")
    channel_name: str = Field(description="Channel or creator name.")
    published_at: str | None = Field(default=None, description="Publication timestamp.")
    duration: str | None = Field(default=None, description="Video duration.")
    view_count: int | None = Field(default=None, description="Total view count.")
    like_count: int | None = Field(default=None, description="Total like count if public.")
