"""LLM Provider Factory for SYNQRA agentic orchestration.

Provides an isolated, provider-agnostic factory for instantiating LangChain chat models
lazily based on SYNQRA settings, supporting structured output, tool binding, clear
configuration error boundaries, and deterministic mock LLMs for testing.
"""

import logging
from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import Field

from synqra.agents.exceptions import LLMConfigurationError, LLMProviderError
from synqra.agents.schemas import LLMProvider
from synqra.core.config import Settings, get_settings

logger = logging.getLogger("synqra.agents.llm")


class MockChatModel(BaseChatModel):
    """Deterministic mock chat model for testing agent workflows without network calls."""

    model_name: str = Field(default="mock-model", description="Identifier for the mock model.")
    mock_responses: list[Any] = Field(
        default_factory=list,
        description="Queue of responses (str, BaseMessage, or dict) to return sequentially.",
    )
    structured_response: Any | None = Field(
        default=None,
        description="Pre-configured response to return when with_structured_output is invoked.",
    )
    recorded_calls: list[list[BaseMessage]] = Field(
        default_factory=list,
        description="Log of input message histories passed to this model.",
    )
    bound_tools: list[Any] = Field(
        default_factory=list,
        description="Tools registered via bind_tools.",
    )

    @property
    def _llm_type(self) -> str:
        return "synqra-mock-chat"

    def _generate(
        self,
        messages: list[BaseMessage],
        _stop: list[str] | None = None,
        _run_manager: CallbackManagerForLLMRun | None = None,
        **_kwargs: Any,
    ) -> ChatResult:
        self.recorded_calls.append(list(messages))

        if self.mock_responses:
            item = self.mock_responses.pop(0)
            if isinstance(item, BaseMessage):
                return ChatResult(generations=[ChatGeneration(message=item)])
            if isinstance(item, str):
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=item))])
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=str(item)))])

        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content=f"Mock response from {self.model_name}"))
            ]
        )

    def with_structured_output(self, schema: Any, **_kwargs: Any) -> Runnable[Any, Any]:
        """Return a runnable that produces structured output conforming to the schema."""

        def _output_resolver(_input_data: Any) -> Any:
            if self.structured_response is not None:
                return self.structured_response

            # If the schema is a Pydantic model or callable, attempt basic instantiation
            if hasattr(schema, "model_validate"):
                try:
                    return schema.model_validate({})
                except Exception:
                    pass
            if callable(schema):
                try:
                    return schema()
                except Exception:
                    pass
            return {"mock": "structured_data"}

        return RunnableLambda(_output_resolver)

    def bind_tools(
        self,
        tools: Sequence[Any],
        **_kwargs: Any,
    ) -> "MockChatModel":
        """Record bound tools and return self for chaining."""
        self.bound_tools.extend(tools)
        return self


def get_mock_llm(
    responses: list[Any] | None = None,
    structured_response: Any | None = None,
    model_name: str = "mock-model",
) -> MockChatModel:
    """Convenience helper to construct a configured MockChatModel for tests."""
    return MockChatModel(
        model_name=model_name,
        mock_responses=list(responses) if responses else [],
        structured_response=structured_response,
    )


def get_llm(
    settings: Settings | None = None,
    provider: str | LLMProvider | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    **kwargs: Any,
) -> BaseChatModel:
    """Initialize and return a LangChain ChatModel instance lazily based on configuration.

    Args:
        settings: Optional Settings instance. Falls back to get_settings() if omitted.
        provider: Provider name ('openai', 'anthropic', 'google', 'mock'). Defaults to settings.
        model: Specific model name to instantiate. Defaults to settings.llm_model.
        temperature: Sampling temperature (default 0.0 for deterministic agent planning).
        **kwargs: Additional provider-specific kwargs passed to model constructor.

    Returns:
        Configured BaseChatModel instance.

    Raises:
        LLMConfigurationError: If required API keys or settings are missing.
        LLMProviderError: If the requested provider is unsupported or uninstalled.
    """
    cfg = settings or get_settings()

    raw_provider = (
        provider.value if isinstance(provider, LLMProvider) else (provider or cfg.llm_provider)
    )
    provider_str = raw_provider.strip().lower()
    model_name = model or cfg.llm_model

    if provider_str == "mock":
        return MockChatModel(model_name=model_name, **kwargs)

    if provider_str == "openai":
        if not cfg.openai_api_key or not cfg.openai_api_key.get_secret_value().strip():
            raise LLMConfigurationError(
                "OpenAI API key is missing. Set SYNQRA_OPENAI_API_KEY or OPENAI_API_KEY "
                "in your environment or .env file."
            )

        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise LLMProviderError(
                "Package 'langchain-openai' is required for the OpenAI provider. "
                "Install it with 'uv add langchain-openai'."
            ) from exc

        return ChatOpenAI(
            api_key=cfg.openai_api_key.get_secret_value(),
            model=model_name,
            temperature=temperature,
            **kwargs,
        )

    if provider_str == "anthropic":
        raise LLMProviderError(
            "Anthropic provider is not yet configured. "
            "Install langchain-anthropic and set ANTHROPIC_API_KEY when enabling Anthropic support."
        )

    if provider_str in ("google", "gemini"):
        raise LLMProviderError(
            "Google Gemini provider is not yet configured. "
            "Install langchain-google-genai and set GOOGLE_API_KEY when enabling Google support."
        )

    raise LLMProviderError(
        f"Unsupported LLM provider '{raw_provider}'. "
        f"Supported providers are: 'openai', 'anthropic', 'google', 'mock'."
    )
