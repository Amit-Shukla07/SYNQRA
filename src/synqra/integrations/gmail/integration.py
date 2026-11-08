"""Gmail integration connector for SYNQRA.

Encapsulates email searching, message inspection, label retrieval, draft creation,
email transmission with confirmation gates, and thread replies via Composio's session-based API.
"""

import logging

from fastmcp import FastMCP

from synqra.core.config import Settings, get_settings
from synqra.integrations.base import BaseIntegration
from synqra.integrations.gmail.client import GmailClient
from synqra.integrations.gmail.schemas import (
    CreateGmailDraftOutput,
    GmailMessageDetails,
    ListGmailLabelsOutput,
    ReplyToGmailMessageOutput,
    SearchGmailMessagesOutput,
    SendGmailMessageOutput,
)
from synqra.integrations.gmail.tools import (
    create_gmail_draft,
    get_gmail_message,
    list_gmail_labels,
    reply_to_gmail_message,
    search_gmail_messages,
    send_gmail_message,
)
from synqra.integrations.models import (
    IntegrationMetadata,
    IntegrationState,
    IntegrationStatus,
    utc_now,
)

logger = logging.getLogger("synqra.integrations.gmail")


class GmailIntegration(BaseIntegration):
    """Production integration connector for Gmail via Composio."""

    def __init__(
        self,
        client: GmailClient | None = None,
        settings: Settings | None = None,
        user_id: str | None = None,
    ) -> None:
        """Initialize the Gmail integration instance.

        Args:
            client: Optional pre-configured GmailClient instance.
            settings: Optional application settings.
            user_id: Optional user identifier for Composio session scoping.
        """
        super().__init__(
            metadata=IntegrationMetadata(
                name="gmail",
                display_name="Gmail",
                description=(
                    "Search emails, retrieve messages, list labels, create drafts, "
                    "send emails with confirmation, and reply to threads."
                ),
                version="0.1.0",
                auth_required=True,
                auth_type="composio",
                documentation_url="https://developers.google.com/gmail",
            )
        )
        self._settings = settings or get_settings()
        self.client = client or GmailClient(
            settings=self._settings,
            user_id=user_id or self._settings.composio_user_id,
        )

    def is_configured(self) -> bool:
        """Check whether Composio credentials are configured."""
        if getattr(self.client, "is_configured", False):
            return True
        return self._settings.composio_api_key is not None

    async def test_connection(self) -> bool:
        """Verify that a valid, active Gmail connection exists through Composio."""
        if not self.is_configured():
            return False
        return await self.client.check_connection_async()

    async def health_check(self) -> IntegrationStatus:
        """Return operational health status, providing an authorization URL if unauthenticated."""
        if not self.is_configured():
            return IntegrationStatus(
                name=self.name,
                display_name=self.display_name,
                state=IntegrationState.UNCONFIGURED,
                is_configured=False,
                is_connected=False,
                registered_tools=[],
                error_message="Composio API key is not configured.",
                last_checked_at=utc_now(),
            )

        try:
            is_connected = await self.test_connection()
            if is_connected:
                return IntegrationStatus(
                    name=self.name,
                    display_name=self.display_name,
                    state=IntegrationState.CONNECTED,
                    is_configured=True,
                    is_connected=True,
                    registered_tools=self.registered_tools,
                    last_checked_at=utc_now(),
                )

            # Not connected: attempt to generate authorization link
            auth_url: str | None = None
            try:
                auth_url = await self.client.authorize_async()
            except Exception as auth_exc:
                logger.debug("Could not generate Gmail auth URL: %s", auth_exc)

            auth_msg = (
                f"Gmail OAuth not connected. Authorize at: {auth_url}"
                if auth_url
                else "Gmail OAuth not connected. Connect account in Composio."
            )

            return IntegrationStatus(
                name=self.name,
                display_name=self.display_name,
                state=IntegrationState.ERROR,
                is_configured=True,
                is_connected=False,
                registered_tools=self.registered_tools,
                error_message=auth_msg,
                last_checked_at=utc_now(),
            )
        except Exception as exc:
            return IntegrationStatus(
                name=self.name,
                display_name=self.display_name,
                state=IntegrationState.ERROR,
                is_configured=True,
                is_connected=False,
                registered_tools=self.registered_tools,
                error_message=f"Health probe failed: {exc}",
                last_checked_at=utc_now(),
            )

    def _register_tools_impl(self, server: FastMCP) -> list[str]:
        """Register the 6 curated Gmail tools on the FastMCP server."""
        client = self.client

        @server.tool(
            name="search_gmail_messages",
            description=(
                "Search Gmail messages using text queries, sender/recipient filters, "
                "subject lines, date ranges, and labels. Returns normalized summaries."
            ),
        )
        async def search_gmail_messages_tool(
            query: str | None = None,
            sender: str | None = None,
            recipient: str | None = None,
            subject: str | None = None,
            keywords: str | None = None,
            after: str | None = None,
            before: str | None = None,
            label_ids: list[str] | None = None,
            max_results: int = 10,
            page_token: str | None = None,
            include_spam_trash: bool = False,
        ) -> SearchGmailMessagesOutput:
            """Search Gmail messages."""
            return await search_gmail_messages(
                client=client,
                query=query,
                sender=sender,
                recipient=recipient,
                subject=subject,
                keywords=keywords,
                after=after,
                before=before,
                label_ids=label_ids,
                max_results=max_results,
                page_token=page_token,
                include_spam_trash=include_spam_trash,
            )

        @server.tool(
            name="get_gmail_message",
            description=(
                "Retrieve complete details for a specific Gmail email using its message ID. "
                "Includes headers, plain text and HTML bodies, and attachment metadata."
            ),
        )
        async def get_gmail_message_tool(
            message_id: str,
            format: str = "full",
        ) -> GmailMessageDetails:
            """Retrieve an individual Gmail message."""
            return await get_gmail_message(
                client=client,
                message_id=message_id,
                format=format,
            )

        @server.tool(
            name="list_gmail_labels",
            description=(
                "List all system and user-defined Gmail labels in the user's mailbox, "
                "including unread counts and total message counts."
            ),
        )
        async def list_gmail_labels_tool(
            include_details: bool = True,
        ) -> ListGmailLabelsOutput:
            """List mailbox labels."""
            return await list_gmail_labels(
                client=client,
                include_details=include_details,
            )

        @server.tool(
            name="create_gmail_draft",
            description=(
                "Create a new email draft in Gmail without sending it. Supports recipients, "
                "CC, BCC, subject line, plain text or HTML bodies, and threading."
            ),
        )
        async def create_gmail_draft_tool(
            recipient: str,
            subject: str,
            body: str,
            cc: list[str] | None = None,
            bcc: list[str] | None = None,
            is_html: bool = False,
            thread_id: str | None = None,
        ) -> CreateGmailDraftOutput:
            """Create an email draft."""
            return await create_gmail_draft(
                client=client,
                recipient=recipient,
                subject=subject,
                body=body,
                cc=cc,
                bcc=bcc,
                is_html=is_html,
                thread_id=thread_id,
            )

        @server.tool(
            name="send_gmail_message",
            description=(
                "Send an email immediately via Gmail. For safety, this external transmission "
                "requires explicit confirmation (confirm=True). If confirm is False, the email "
                "will not be sent."
            ),
        )
        async def send_gmail_message_tool(
            recipient: str,
            subject: str,
            body: str,
            cc: list[str] | None = None,
            bcc: list[str] | None = None,
            is_html: bool = False,
            confirm: bool = False,
        ) -> SendGmailMessageOutput:
            """Send an email with mandatory confirmation safety check."""
            return await send_gmail_message(
                client=client,
                recipient=recipient,
                subject=subject,
                body=body,
                cc=cc,
                bcc=bcc,
                is_html=is_html,
                confirm=confirm,
            )

        @server.tool(
            name="reply_to_gmail_message",
            description=(
                "Reply to an existing email message, preserving thread context and headers. "
                "For safety, sending a reply requires explicit confirmation (confirm=True)."
            ),
        )
        async def reply_to_gmail_message_tool(
            message_id: str,
            reply_body: str,
            thread_id: str | None = None,
            recipient: str | None = None,
            cc: list[str] | None = None,
            bcc: list[str] | None = None,
            is_html: bool = False,
            confirm: bool = False,
        ) -> ReplyToGmailMessageOutput:
            """Reply to an email thread with mandatory confirmation safety check."""
            return await reply_to_gmail_message(
                client=client,
                message_id=message_id,
                reply_body=reply_body,
                thread_id=thread_id,
                recipient=recipient,
                cc=cc,
                bcc=bcc,
                is_html=is_html,
                confirm=confirm,
            )

        return [
            "search_gmail_messages",
            "get_gmail_message",
            "list_gmail_labels",
            "create_gmail_draft",
            "send_gmail_message",
            "reply_to_gmail_message",
        ]
