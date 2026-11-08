"""Standardized exception hierarchy for SYNQRA agentic orchestration.

Provides clear contextual error handling for planning, LLM provider initialization,
state validation, dependency graph resolution, and human approvals.
"""


class SynqraAgentError(Exception):
    """Base exception for all agent orchestration errors in SYNQRA."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class LLMConfigurationError(SynqraAgentError):
    """Raised when LLM configuration or required API keys are missing or invalid."""


class LLMProviderError(SynqraAgentError):
    """Raised when an unsupported or unconfigured LLM provider is requested."""


class InvalidExecutionPlanError(SynqraAgentError):
    """Raised when an execution plan violates dependency or validation constraints."""


class InvalidAgentStateError(SynqraAgentError):
    """Raised when an agent state transition or operation violates consistency rules."""


class ApprovalError(SynqraAgentError):
    """Raised when an approval request or decision is invalid or unauthorized."""


class CheckpointerError(SynqraAgentError):
    """Base exception for LangGraph checkpointer initialization and operation errors."""


class CheckpointerConfigurationError(CheckpointerError):
    """Raised when checkpointer configuration is missing or invalid."""


class CheckpointerConnectionError(CheckpointerError):
    """Raised when checkpointer database connection fails or times out."""


class WorkflowRecoveryError(SynqraAgentError):
    """Raised when an interrupted workflow cannot be safely recovered or resumed."""
