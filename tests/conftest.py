"""Pytest configuration and test isolation fixtures."""

import pytest

from synqra.core.config import Settings, get_settings


@pytest.fixture(autouse=True)
def isolated_test_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure unit tests run in a completely isolated environment without live credentials."""
    monkeypatch.setattr(
        "synqra.core.config.Settings.model_config",
        {"env_file": None, "case_sensitive": False, "extra": "ignore"},
    )
    monkeypatch.delenv("SYNQRA_COMPOSIO_API_KEY", raising=False)
    monkeypatch.delenv("COMPOSIO_API_KEY", raising=False)
    monkeypatch.delenv("SYNQRA_YOUTUBE_API_KEY", raising=False)
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    monkeypatch.delenv("SYNQRA_GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("SYNQRA_TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("SYNQRA_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SYNQRA_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("SYNQRA_LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    get_settings.cache_clear()

    # Re-register YouTube, GitHub, Google Drive, Google Calendar, Gmail, and Web Search
    # integrations in default_registry with clean unconfigured settings
    from synqra.integrations import default_registry
    from synqra.integrations.github import GitHubIntegration
    from synqra.integrations.gmail import GmailIntegration
    from synqra.integrations.google_calendar import GoogleCalendarIntegration
    from synqra.integrations.google_drive import GoogleDriveIntegration
    from synqra.integrations.web_search import WebSearchIntegration
    from synqra.integrations.youtube import YouTubeIntegration

    clean_settings = Settings()
    default_registry.register(YouTubeIntegration(settings=clean_settings))
    default_registry.register(GitHubIntegration(settings=clean_settings))
    default_registry.register(GoogleDriveIntegration(settings=clean_settings))
    default_registry.register(GoogleCalendarIntegration(settings=clean_settings))
    default_registry.register(GmailIntegration(settings=clean_settings))
    default_registry.register(WebSearchIntegration(settings=clean_settings))

    yield

    get_settings.cache_clear()
