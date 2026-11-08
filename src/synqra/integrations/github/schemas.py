"""Pydantic schemas for the GitHub integration in SYNQRA.

Defines validated input parameters and structured response models for repository search,
metadata retrieval, file contents, issues, and README extraction.
"""

from typing import Literal

from pydantic import BaseModel, Field


class GitHubRepositorySummary(BaseModel):
    """Normalized summary of a GitHub repository."""

    full_name: str = Field(description="Full repository identifier, e.g. 'owner/repo'.")
    name: str = Field(description="Repository name.")
    owner: str = Field(description="Account or organization owner.")
    description: str | None = Field(default=None, description="Repository description.")
    url: str = Field(description="Web URL of the repository on GitHub.")
    stars: int = Field(default=0, description="Number of stargazers.")
    forks: int = Field(default=0, description="Number of forks.")
    language: str | None = Field(default=None, description="Primary programming language.")
    topics: list[str] = Field(default_factory=list, description="Repository topics or tags.")
    created_at: str | None = Field(
        default=None, description="ISO timestamp of repository creation."
    )
    updated_at: str | None = Field(
        default=None, description="ISO timestamp of last repository update."
    )


class GitHubSearchRepositoriesInput(BaseModel):
    """Input parameters for searching GitHub repositories."""

    query: str = Field(
        description="Search keyword, repo name, or topic qualifier (e.g. 'machine learning').",
        min_length=1,
    )
    language: str | None = Field(
        default=None,
        description="Filter by primary language (e.g. 'python', 'rust', 'typescript').",
    )
    sort: Literal["stars", "forks", "help-wanted-issues", "updated"] | None = Field(
        default="stars",
        description="Sort field for results.",
    )
    order: Literal["asc", "desc"] = Field(
        default="desc",
        description="Sort direction (ascending or descending).",
    )
    page: int = Field(default=1, ge=1, description="Page number of results to fetch.")
    per_page: int = Field(default=10, ge=1, le=100, description="Results per page (1-100).")


class GitHubSearchRepositoriesOutput(BaseModel):
    """Structured response containing repository search results."""

    query: str = Field(description="The submitted search query.")
    total_count: int = Field(description="Total repositories found.")
    page: int = Field(description="Current page returned.")
    repositories: list[GitHubRepositorySummary] = Field(
        default_factory=list,
        description="List of matching repositories.",
    )


class GitHubGetRepositoryInput(BaseModel):
    """Input parameters for fetching detailed repository information."""

    owner: str = Field(description="GitHub username or organization name.", min_length=1)
    repo: str = Field(description="Repository name without .git suffix.", min_length=1)


class GitHubRepositoryDetails(BaseModel):
    """Detailed metadata for an individual GitHub repository."""

    full_name: str = Field(description="Full name, e.g. 'owner/repo'.")
    name: str = Field(description="Repository name.")
    owner: str = Field(description="Owner username or organization.")
    description: str | None = Field(default=None, description="Repository description.")
    url: str = Field(description="Web URL of the repository.")
    stars: int = Field(default=0, description="Star count.")
    forks: int = Field(default=0, description="Fork count.")
    language: str | None = Field(default=None, description="Primary programming language.")
    topics: list[str] = Field(default_factory=list, description="Repository topics.")
    created_at: str | None = Field(default=None, description="Creation timestamp.")
    updated_at: str | None = Field(default=None, description="Last update timestamp.")
    default_branch: str = Field(default="main", description="Default branch name.")
    open_issues_count: int = Field(default=0, description="Count of open issues and PRs.")
    license_name: str | None = Field(default=None, description="SPDX identifier or license name.")


class GitHubGetFileContentsInput(BaseModel):
    """Input parameters for retrieving file contents from a repository."""

    owner: str = Field(description="Repository owner (username or organization).", min_length=1)
    repo: str = Field(description="Repository name.", min_length=1)
    path: str = Field(
        description="Relative path to file in repository (e.g. 'README.md', 'src/main.py')."
    )
    ref: str | None = Field(
        default=None,
        description="Optional Git reference (branch name, tag name, or commit SHA).",
    )


class GitHubFileContentsOutput(BaseModel):
    """Normalized response containing file contents and metadata."""

    owner: str = Field(description="Repository owner.")
    repo: str = Field(description="Repository name.")
    path: str = Field(description="Path of the file.")
    ref: str | None = Field(default=None, description="Resolved Git reference or branch.")
    content: str | None = Field(default=None, description="Decoded textual content of the file.")
    size: int = Field(default=0, description="File size in bytes.")
    encoding: str = Field(
        default="utf-8", description="Encoding format ('utf-8', 'base64', 'binary')."
    )
    is_binary: bool = Field(default=False, description="Flag indicating if the file is binary.")
    download_url: str | None = Field(default=None, description="Direct download URL if available.")


class GitHubListIssuesInput(BaseModel):
    """Input parameters for listing issues in a repository."""

    owner: str = Field(description="Repository owner (username or organization).", min_length=1)
    repo: str = Field(description="Repository name.", min_length=1)
    state: Literal["open", "closed", "all"] = Field(
        default="open",
        description="Filter by issue state ('open', 'closed', 'all').",
    )
    labels: str | None = Field(
        default=None,
        description="Comma-separated list of label names (e.g. 'bug,help wanted').",
    )
    page: int = Field(default=1, ge=1, description="Page number of issues to fetch.")
    per_page: int = Field(
        default=10, ge=1, le=100, description="Number of issues per page (1-100)."
    )


class GitHubIssueItem(BaseModel):
    """Normalized repository issue item (excluding pull requests)."""

    number: int = Field(description="Issue number.")
    title: str = Field(description="Issue title.")
    state: str = Field(description="Issue state ('open' or 'closed').")
    url: str = Field(description="Web URL of the issue.")
    author: str | None = Field(default=None, description="Author username.")
    labels: list[str] = Field(default_factory=list, description="Associated issue label names.")
    created_at: str | None = Field(default=None, description="Creation timestamp.")
    updated_at: str | None = Field(default=None, description="Last update timestamp.")
    comments_count: int = Field(default=0, description="Number of comments on the issue.")
    body_preview: str | None = Field(default=None, description="Truncated body preview.")


class GitHubListIssuesOutput(BaseModel):
    """Structured response containing filtered repository issues."""

    owner: str = Field(description="Repository owner.")
    repo: str = Field(description="Repository name.")
    page: int = Field(description="Current page.")
    total_returned: int = Field(description="Number of issues returned in this page.")
    issues: list[GitHubIssueItem] = Field(
        default_factory=list,
        description="List of issues (pull requests excluded).",
    )


class GitHubGetReadmeInput(BaseModel):
    """Input parameters for retrieving the repository README."""

    owner: str = Field(description="Repository owner.", min_length=1)
    repo: str = Field(description="Repository name.", min_length=1)
    ref: str | None = Field(
        default=None, description="Optional Git reference (branch or commit SHA)."
    )


class GitHubReadmeOutput(BaseModel):
    """Normalized response containing README content and metadata."""

    owner: str = Field(description="Repository owner.")
    repo: str = Field(description="Repository name.")
    ref: str | None = Field(default=None, description="Git reference or branch.")
    filename: str = Field(default="README.md", description="README file name.")
    content: str = Field(description="Decoded Markdown/text content of the README.")
    size: int = Field(default=0, description="Size in bytes.")
    encoding: str = Field(default="utf-8", description="Content encoding.")
    download_url: str | None = Field(default=None, description="Direct download URL if available.")
