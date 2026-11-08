"""Interactive terminal chat session loop for SYNQRA (Phase 9.6)."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from synqra.cli.commands import display_status, run_agent_request
from synqra.cli.display import (
    console,
    render_banner,
    render_integrations_summary,
)
from synqra.integrations import default_registry

logger = logging.getLogger("synqra.cli.chat")


def get_integrations_summary_list() -> list[dict[str, Any]]:
    """Gather integration names and configuration state for chat banner."""
    available_names = set(default_registry.list_available())
    all_names = [
        ("GitHub", "github"),
        ("Gmail", "gmail"),
        ("Google Drive", "google_drive"),
        ("Google Calendar", "google_calendar"),
        ("YouTube", "youtube"),
        ("Web Search", "web_search"),
    ]
    return [{"name": display, "configured": key in available_names} for display, key in all_names]


def display_chat_help() -> None:
    """Print available interactive commands."""
    console.print(
        "\n[bold cyan]SYNQRA Chat Commands:[/bold cyan]\n"
        "  [bold]/help[/bold]    - Show this help message\n"
        "  [bold]/status[/bold]  - View connected integrations and checkpointer status\n"
        "  [bold]/clear[/bold]   - Reset conversation session and start a new thread\n"
        "  [bold]/exit[/bold]    - End the chat session\n"
        "  [bold]exit[/bold]     - End the chat session\n"
    )


class ChatSession:
    """Manages an interactive multi-turn conversation session with SYNQRA."""

    def __init__(self, llm: Any | None = None, adapter: Any | None = None) -> None:
        self.session_thread_id: str = f"th_chat_{uuid.uuid4().hex[:12]}"
        self.history: list[dict[str, str]] = []
        self._llm = llm
        self._adapter = adapter

    def reset_session(self) -> None:
        """Clear conversation history and allocate a new thread ID."""
        self.session_thread_id = f"th_chat_{uuid.uuid4().hex[:12]}"
        self.history.clear()
        console.print("[dim]Conversation history reset. Started new session thread.[/dim]")

    async def handle_user_message(self, user_text: str) -> None:
        """Process a single turn of user input."""
        clean = user_text.strip()
        if not clean:
            return

        # Check slash commands
        if clean.lower() in ("/exit", "/quit", "exit", "quit"):
            raise EOFError

        if clean.lower() == "/help":
            display_chat_help()
            return

        if clean.lower() == "/status":
            await display_status()
            return

        if clean.lower() == "/clear":
            self.reset_session()
            return

        # Augment natural language request with recent in-session context if available
        if self.history:
            recent_context = "\n".join(
                f"{h['role'].title()}: {h['content']}" for h in self.history[-4:]
            )
            augmented_request = (
                f"Prior conversation context in this session:\n{recent_context}\n\n"
                f"User request: {clean}"
            )
        else:
            augmented_request = clean

        self.history.append({"role": "user", "content": clean})

        resp = await run_agent_request(
            request=augmented_request,
            thread_id=self.session_thread_id,
            llm=self._llm,
            adapter=self._adapter,
            silent=False,
        )

        agent_response_text = resp["result"].get("final_response") or resp["message"]
        self.history.append({"role": "assistant", "content": agent_response_text})


async def run_chat_session(llm: Any | None = None) -> None:
    """Launch and manage the interactive terminal chat experience."""
    render_banner()
    render_integrations_summary(get_integrations_summary_list())
    console.print(
        "[dim]Type your request below. Use [bold]/help[/bold] for commands or "
        "[bold]/exit[/bold] to leave.[/dim]\n"
    )

    session = ChatSession(llm=llm)

    while True:
        try:
            user_input = console.input("[bold cyan]SYNQRA >[/bold cyan] ")
            await session.handle_user_message(user_input)
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold green]Goodbye![/bold green] Thank you for using SYNQRA.\n")
            break
        except Exception as exc:
            logger.error("Chat error: %s", exc)
            console.print(f"[bold red]Unexpected error:[/bold red] {exc}")
