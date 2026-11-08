"""CLI exceptions for SYNQRA (Phase 9.6)."""

from __future__ import annotations


class SynqraCLIError(Exception):
    """Base exception for CLI-related errors."""


class CommandExecutionError(SynqraCLIError):
    """Raised when single-command agent execution fails."""


class ChatSessionError(SynqraCLIError):
    """Raised when an interactive chat session encounters an unrecoverable failure."""
