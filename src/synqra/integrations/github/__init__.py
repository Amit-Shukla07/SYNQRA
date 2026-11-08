"""GitHub integration package for SYNQRA.

Exposes GitHubIntegration and related client/tools.
"""

from synqra.integrations.github.client import GitHubClient
from synqra.integrations.github.integration import GitHubIntegration

__all__ = ["GitHubClient", "GitHubIntegration"]
