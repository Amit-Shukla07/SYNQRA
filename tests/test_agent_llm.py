"""Unit tests for the SYNQRA LLM Provider Factory and MockChatModel."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, SecretStr

from synqra.agents.exceptions import LLMConfigurationError, LLMProviderError
from synqra.agents.llm import MockChatModel, get_llm, get_mock_llm
from synqra.agents.schemas import LLMProvider
from synqra.core.config import Settings


def test_llm_factory_missing_openai_key() -> None:
    """Verify LLMConfigurationError is raised when OpenAI key is missing."""
    settings = Settings()
    # In clean isolated environment, openai_api_key is None
    with pytest.raises(LLMConfigurationError, match="OpenAI API key is missing"):
        get_llm(settings=settings, provider="openai")


def test_llm_factory_openai_instantiation() -> None:
    """Verify ChatOpenAI is instantiated when API key is provided."""
    from langchain_openai import ChatOpenAI

    settings = Settings(openai_api_key=SecretStr("sk-test-mock-key-12345"))
    llm = get_llm(settings=settings, provider="openai")

    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == "gpt-4o-mini"
    assert llm.temperature == 0.0


def test_llm_factory_custom_model_configuration() -> None:
    """Verify custom model name override in get_llm."""
    from langchain_openai import ChatOpenAI

    settings = Settings(openai_api_key=SecretStr("sk-test-mock-key-12345"))
    llm = get_llm(settings=settings, provider=LLMProvider.OPENAI, model="gpt-4o", temperature=0.2)

    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == "gpt-4o"
    assert llm.temperature == 0.2


def test_llm_factory_unsupported_provider() -> None:
    """Verify unsupported provider string raises LLMProviderError."""
    settings = Settings()
    with pytest.raises(LLMProviderError, match="Unsupported LLM provider 'cohere'"):
        get_llm(settings=settings, provider="cohere")


def test_llm_factory_future_providers() -> None:
    """Verify currently unconfigured future providers raise clear LLMProviderErrors."""
    settings = Settings()
    with pytest.raises(LLMProviderError, match="Anthropic provider is not yet configured"):
        get_llm(settings=settings, provider="anthropic")

    with pytest.raises(LLMProviderError, match="Google Gemini provider is not yet configured"):
        get_llm(settings=settings, provider="google")


def test_mock_chat_model_sync_invoke() -> None:
    """Verify MockChatModel returns queued responses synchronously."""
    mock_llm = get_mock_llm(responses=["First plan step", "Second summary response"])

    resp1 = mock_llm.invoke([HumanMessage(content="Hello")])
    assert isinstance(resp1, AIMessage)
    assert resp1.content == "First plan step"

    resp2 = mock_llm.invoke([HumanMessage(content="Next")])
    assert resp2.content == "Second summary response"

    # Fallback default response
    resp3 = mock_llm.invoke([HumanMessage(content="Again")])
    assert "Mock response from mock-model" in resp3.content
    assert len(mock_llm.recorded_calls) == 3


@pytest.mark.asyncio
async def test_mock_chat_model_async_invoke() -> None:
    """Verify MockChatModel works with async ainvoke."""
    mock_llm = get_mock_llm(responses=["Async agent response"])
    res = await mock_llm.ainvoke("Generate plan")
    assert res.content == "Async agent response"


def test_mock_chat_model_structured_output() -> None:
    """Verify MockChatModel with_structured_output returns schema conforming data."""

    class TestOutput(BaseModel):
        summary: str = "Plan summary"
        num_steps: int = 4

    # 1. Preset structured response
    expected = TestOutput(summary="Custom plan", num_steps=2)
    mock_llm = get_mock_llm(structured_response=expected)
    runnable = mock_llm.with_structured_output(TestOutput)
    out = runnable.invoke("Create plan")
    assert out == expected

    # 2. Schema default instantiation
    mock_llm_default = MockChatModel()
    runnable_default = mock_llm_default.with_structured_output(TestOutput)
    out_default = runnable_default.invoke("Create plan")
    assert isinstance(out_default, TestOutput)
    assert out_default.summary == "Plan summary"


def test_mock_chat_model_tool_binding() -> None:
    """Verify MockChatModel records bound tools."""
    mock_llm = MockChatModel()
    dummy_tools = [{"name": "search_web"}, {"name": "search_github"}]
    bound = mock_llm.bind_tools(dummy_tools)
    assert bound.bound_tools == dummy_tools
