"""Unit and integration tests for SYNQRA CLI (Phase 9.6 Component A)."""

from __future__ import annotations

from typing import Any

import pytest
from click.testing import CliRunner

from synqra.agents.llm import MockChatModel
from synqra.agents.planner import PlannerOutput, PlanStepProposal
from synqra.agents.tool_adapter import AgentToolAdapter
from synqra.cli.chat import ChatSession
from synqra.cli.commands import run_agent_request
from synqra.cli.main import app


class DummyAdapter(AgentToolAdapter):
    """Mock adapter returning successful read outputs for CLI unit tests."""

    async def execute(
        self,
        tool: Any = None,
        arguments: dict[str, Any] | None = None,
        timeout: float | None = None,
        **_kwargs: Any,
    ) -> Any:
        _ = (tool, timeout)
        args = arguments or {}
        return {"query": args.get("query"), "results": ["FastAPI docs found."]}


def test_cli_version() -> None:
    """Verify `synqra version` outputs application version."""
    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "SYNQRA version 0.1.0" in result.output


def test_cli_help() -> None:
    """Verify `synqra --help` displays all expected commands."""
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "chat" in result.output
    assert "run" in result.output
    assert "server" in result.output
    assert "status" in result.output
    assert "version" in result.output


def test_cli_status() -> None:
    """Verify `synqra status` displays the rich status table."""
    runner = CliRunner()
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "SYNQRA System Status" in result.output
    assert "Core Server" in result.output
    assert "Transport" in result.output
    assert "Checkpointer" in result.output


@pytest.mark.asyncio
async def test_cli_run_empty_request_rejected() -> None:
    """Verify empty natural language requests are rejected cleanly."""
    res = await run_agent_request("", silent=True)
    assert res["status"] == "failed"
    assert "cannot be empty" in res["message"].lower()


@pytest.mark.asyncio
async def test_cli_run_single_command_read_success() -> None:
    """Verify single-shot agent execution on read-only queries."""
    llm = MockChatModel(
        structured_response=PlannerOutput(
            reasoning_summary="Search web for FastAPI documentation",
            steps=[
                PlanStepProposal(
                    step_id="step_1",
                    tool_name="search_web",
                    arguments={"query": "FastAPI documentation"},
                    description="Search web",
                )
            ],
        ),
    )

    res = await run_agent_request(
        request="Search web for FastAPI documentation",
        llm=llm,
        adapter=DummyAdapter(),
        silent=True,
    )

    assert res["status"] in ("completed", "planning")
    assert res["workflow_id"].startswith("wf_")
    assert res["thread_id"].startswith("th_")
    assert res["approval_required"] is False
    assert len(res["pending_approvals"]) == 0


@pytest.mark.asyncio
async def test_cli_run_approval_required_interruption() -> None:
    """Verify single-shot agent execution pauses on protected write operations."""
    llm = MockChatModel(
        structured_response=PlannerOutput(
            reasoning_summary="Send an email to user",
            steps=[
                PlanStepProposal(
                    step_id="step_1",
                    tool_name="send_gmail_message",
                    arguments={"recipient": "user@example.com", "subject": "Hi", "body": "Hello"},
                    description="Send notification email",
                )
            ],
        ),
    )

    res = await run_agent_request(
        request="Send an email to user@example.com",
        llm=llm,
        silent=True,
    )

    assert res["status"] == "waiting_for_approval"
    assert res["approval_required"] is True
    assert len(res["pending_approvals"]) == 1

    appr = res["pending_approvals"][0]
    assert appr["tool_name"] == "send_gmail_message"
    assert appr["arguments"]["recipient"] == "user@example.com"
    assert appr["permission_level"] in ("write", "sensitive")
    assert appr["approval_id"].startswith("appr_")


@pytest.mark.asyncio
async def test_cli_chat_session_slash_commands() -> None:
    """Verify slash commands (/help, /status, /clear, /exit) in chat session."""
    session = ChatSession()

    # /help does not raise
    await session.handle_user_message("/help")

    # /status does not raise
    await session.handle_user_message("/status")

    # /clear resets session
    old_thread = session.session_thread_id
    session.history.append({"role": "user", "content": "hi"})
    await session.handle_user_message("/clear")
    assert session.session_thread_id != old_thread
    assert len(session.history) == 0

    # /exit raises EOFError to cleanly terminate loop
    with pytest.raises(EOFError):
        await session.handle_user_message("/exit")


@pytest.mark.asyncio
async def test_cli_chat_session_multi_turn_context() -> None:
    """Verify consecutive messages in a chat session preserve thread ID and history."""
    llm = MockChatModel(
        structured_response=PlannerOutput(
            reasoning_summary="Retrieve video details",
            steps=[
                PlanStepProposal(
                    step_id="step_1",
                    tool_name="search_youtube_videos",
                    arguments={"query": "Python tutorial"},
                    description="Search youtube",
                )
            ],
        ),
    )
    session = ChatSession(llm=llm, adapter=DummyAdapter())

    # Turn 1
    await session.handle_user_message("Search for Python tutorials")
    assert len(session.history) == 2
    assert session.history[0]["role"] == "user"
    assert session.history[1]["role"] == "assistant"

    # Turn 2
    tid_turn1 = session.session_thread_id
    await session.handle_user_message("Which one is the best?")
    assert len(session.history) == 4
    # Reuses same session thread ID across turns
    assert session.session_thread_id == tid_turn1


class MultiTurnGitHubAdapter(AgentToolAdapter):
    """Mock adapter returning GitHub responses."""

    def __init__(self) -> None:
        self.invocations: list[dict[str, Any]] = []

    async def execute(
        self,
        tool: Any = None,
        arguments: dict[str, Any] | None = None,
        timeout: float | None = None,
        **_kwargs: Any,
    ) -> Any:
        _ = timeout
        args = arguments or {}
        tool_name = getattr(tool, "name", str(tool))
        self.invocations.append({"tool": tool_name, "arguments": args})
        if tool_name == "search_github_repositories":
            return {
                "total_count": 3,
                "repositories": [
                    {"name": "repo-alpha", "owner": "octocat", "stargazers_count": 12},
                    {"name": "star-champion", "owner": "octocat", "stargazers_count": 450},
                    {"name": "repo-beta", "owner": "octocat", "stargazers_count": 5},
                ],
            }
        if tool_name == "get_github_readme":
            return {
                "owner": args.get("owner"),
                "repo": args.get("repo"),
                "content": "# star-champion\nThis repository has 450 stars.",
            }
        return {}


class ThreeTurnGitHubLLM(MockChatModel):
    """Mock LLM handling multi-turn planner and aggregator outputs."""

    def _generate(
        self,
        messages: Any,
        _stop: Any = None,
        _run_manager: Any = None,
        **_kwargs: Any,
    ) -> Any:
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        prompt_text = "\n".join(getattr(m, "content", "") for m in messages)
        if "Show me its README" in prompt_text:
            content = "# star-champion README content with 450 stars."
        elif "Which one has the most stars" in prompt_text:
            content = "star-champion has the most stars with 450 stars."
        else:
            content = (
                "Found 3 repositories: star-champion (450 stars), repo-alpha (12 stars), "
                "repo-beta (5 stars)."
            )

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    def with_structured_output(self, schema: Any, **_kwargs: Any) -> Any:
        from langchain_core.runnables import RunnableLambda

        _ = schema

        def _resolve(messages: Any) -> Any:
            prompt_text = "\n".join(getattr(m, "content", "") for m in messages)

            # Turn 3: Show me its README. Must resolve "its" from prior context!
            if "Show me its README" in prompt_text:
                assert "star-champion" in prompt_text, "Context missing star-champion!"
                assert "octocat" in prompt_text or "450" in prompt_text
                return PlannerOutput(
                    reasoning_summary="Retrieve README for star-champion repository",
                    steps=[
                        PlanStepProposal(
                            step_id="step_1",
                            tool_name="get_github_readme",
                            arguments={"owner": "octocat", "repo": "star-champion"},
                            description="Fetch README for star-champion",
                        )
                    ],
                )

            # Turn 2: Which one has the most stars?
            if "Which one has the most stars" in prompt_text:
                assert "star-champion" in prompt_text, "Context missing star-champion!"
                assert "450" in prompt_text, "Context missing star count!"
                return PlannerOutput(
                    reasoning_summary="star-champion has the highest star count with 450 stars.",
                    steps=[],
                )

            # Turn 1: Find my latest GitHub repositories
            return PlannerOutput(
                reasoning_summary="Search GitHub repositories for current user",
                steps=[
                    PlanStepProposal(
                        step_id="step_1",
                        tool_name="search_github_repositories",
                        arguments={"query": "user:octocat"},
                        description="List octocat repos",
                    )
                ],
            )

        return RunnableLambda(_resolve)


@pytest.mark.asyncio
async def test_cli_chat_three_turn_github_context_resolution() -> None:
    """Verify three-turn conversation context preservation, reference resolution, and isolation.

    Turn 1: 'Find my latest GitHub repositories.' -> executes search_github_repositories
    Turn 2: 'Which one has the most stars?' -> resolves using prior conversation context
    Turn 3: 'Show me its README.' -> resolves 'its' to 'star-champion' and calls get_github_readme
    """
    adapter = MultiTurnGitHubAdapter()
    llm = ThreeTurnGitHubLLM()
    session = ChatSession(llm=llm, adapter=adapter)
    original_thread_id = session.session_thread_id

    # Turn 1: Find my latest GitHub repositories.
    await session.handle_user_message("Find my latest GitHub repositories.")
    assert len(session.history) == 2
    assert session.session_thread_id == original_thread_id
    assert "star-champion" in session.history[1]["content"]
    assert any(inv["tool"] == "search_github_repositories" for inv in adapter.invocations)

    # Turn 2: Which one has the most stars?
    await session.handle_user_message("Which one has the most stars?")
    assert len(session.history) == 4
    assert session.session_thread_id == original_thread_id
    assert "star-champion" in session.history[3]["content"]

    # Turn 3: Show me its README.
    await session.handle_user_message("Show me its README.")
    assert len(session.history) == 6
    assert session.session_thread_id == original_thread_id
    # Verify get_github_readme was called with resolved arguments owner=octocat, repo=star-champion
    readme_invocations = [inv for inv in adapter.invocations if inv["tool"] == "get_github_readme"]
    assert len(readme_invocations) == 1
    assert readme_invocations[0]["arguments"] == {"owner": "octocat", "repo": "star-champion"}
    assert "README" in session.history[5]["content"]

    # Verify context does NOT leak to a new chat session
    session2 = ChatSession()
    assert session2.session_thread_id != session.session_thread_id
    assert len(session2.history) == 0

    # Verify /clear resets context and allocates a new thread ID
    await session.handle_user_message("/clear")
    assert len(session.history) == 0
    assert session.session_thread_id != original_thread_id
