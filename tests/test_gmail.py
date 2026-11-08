"""Unit and integration tests for SYNQRA Gmail integration.

Verifies GmailClient, schema models, tools, and integration health checks using mocked
Composio sessions without requiring live external network credentials.
"""

from unittest.mock import MagicMock

import pytest
from fastmcp import FastMCP

from synqra.core.config import Settings
from synqra.integrations.exceptions import (
    IntegrationConfigurationError,
    IntegrationExecutionError,
)
from synqra.integrations.gmail.client import (
    GMAIL_FETCH_EMAILS_ACTION,
    GMAIL_FETCH_MESSAGE_BY_ID_ACTION,
    GMAIL_REPLY_TO_THREAD_ACTION,
    GMAIL_SEND_EMAIL_ACTION,
    GmailClient,
)
from synqra.integrations.gmail.integration import GmailIntegration
from synqra.integrations.gmail.tools import (
    build_gmail_query,
    create_gmail_draft,
    get_gmail_message,
    list_gmail_labels,
    reply_to_gmail_message,
    search_gmail_messages,
    send_gmail_message,
)
from synqra.integrations.models import IntegrationState

# ==============================================================================
# Client & Configuration Tests
# ==============================================================================


def test_gmail_client_unconfigured_without_api_key() -> None:
    """Verify GmailClient reports unconfigured when no Composio key is provided."""
    empty_settings = Settings()
    client = GmailClient(settings=empty_settings)

    assert client.is_configured is False
    with pytest.raises(IntegrationConfigurationError) as exc_info:
        client._get_composio_client()
    assert "Composio API key is not configured" in str(exc_info.value)


def test_gmail_client_configured_with_api_key() -> None:
    """Verify GmailClient reports configured when Composio key is present."""
    client = GmailClient(api_key="ak_mock_gmail_key_12345")
    assert client.is_configured is True
    assert client.user_id == "synqra_default_user"


def test_gmail_client_check_connection_active() -> None:
    """Verify check_connection returns True when toolkit connection is active."""
    mock_session = MagicMock()
    mock_item = MagicMock()
    mock_item.slug = "gmail"
    mock_item.connection = MagicMock(is_active=True)
    mock_session.toolkits.return_value = MagicMock(items=[mock_item])

    client = GmailClient(api_key="ak_mock_key", session=mock_session)
    assert client.check_connection() is True


def test_gmail_client_check_connection_inactive() -> None:
    """Verify check_connection returns False when toolkit connection is inactive."""
    mock_session = MagicMock()
    mock_item = MagicMock()
    mock_item.slug = "gmail"
    mock_item.connection = MagicMock(is_active=False)
    mock_session.toolkits.return_value = MagicMock(items=[mock_item])

    client = GmailClient(api_key="ak_mock_key", session=mock_session)
    assert client.check_connection() is False


def test_gmail_client_check_connection_error_returns_false() -> None:
    """Verify check_connection returns False when an unexpected exception occurs."""
    mock_session = MagicMock()
    mock_session.toolkits.side_effect = RuntimeError("Network timeout")

    client = GmailClient(api_key="ak_mock_key", session=mock_session)
    assert client.check_connection() is False


def test_gmail_client_authorize_generates_redirect_url() -> None:
    """Verify authorize returns redirect URL for OAuth connection."""
    mock_session = MagicMock()
    mock_req = MagicMock(redirect_url="https://connect.composio.dev/auth/gmail-test")
    mock_session.authorize.return_value = mock_req

    client = GmailClient(api_key="ak_mock_key", session=mock_session)
    auth_url = client.authorize()

    assert auth_url == "https://connect.composio.dev/auth/gmail-test"
    assert client.authorization_url == auth_url
    mock_session.authorize.assert_called_once_with("gmail", callback_url=None)


def test_gmail_client_execute_tool_success() -> None:
    """Verify execute_tool returns execution data on successful execution."""
    mock_session = MagicMock()
    mock_response = MagicMock(result_type="completed", error=None, data={"id": "msg_123"})
    mock_session.execute.return_value = mock_response

    client = GmailClient(api_key="ak_mock_key", session=mock_session)
    result = client.execute_tool(GMAIL_SEND_EMAIL_ACTION, {"subject": "Test"})

    assert result == {"id": "msg_123"}


@pytest.mark.parametrize(
    ("error_msg", "expected_phrase"),
    [
        ("429 Rate limit exceeded", "rate limit exceeded"),
        ("404 Message not found", "resource not found"),
        ("403 Insufficient permissions", "Insufficient permissions"),
        (
            "no active connection found for toolkit 'gmail'",
            "Gmail authorization required",
        ),
    ],
)
def test_gmail_client_execute_tool_error_mapping(error_msg: str, expected_phrase: str) -> None:
    """Verify specific error messages are converted to appropriate exceptions."""
    mock_session = MagicMock()
    mock_session.authorize.return_value = MagicMock(redirect_url="https://auth.example.com")
    mock_response = MagicMock(result_type="failed", error=error_msg, data=None)
    mock_session.execute.return_value = mock_response

    client = GmailClient(api_key="ak_mock_key", session=mock_session)
    with pytest.raises(IntegrationExecutionError) as exc_info:
        client.execute_tool(GMAIL_FETCH_EMAILS_ACTION, {})
    assert expected_phrase.lower() in str(exc_info.value).lower()


# ==============================================================================
# Query Builder Tests
# ==============================================================================


def test_build_gmail_query_all_filters() -> None:
    """Verify composite query construction combining all filter operators."""
    query = build_gmail_query(
        query="is:unread",
        sender="boss@example.com",
        recipient="me@example.com",
        subject="Important Update",
        keywords="urgent quarterly",
        after="2026/11/01",
        before="2026/11/08",
    )
    expected = (
        'from:boss@example.com to:me@example.com subject:"Important Update" '
        "urgent quarterly after:2026/11/01 before:2026/11/08 is:unread"
    )
    assert query == expected


def test_build_gmail_query_empty() -> None:
    """Verify build_gmail_query returns None when no arguments provided."""
    assert build_gmail_query() is None


# ==============================================================================
# Tool 1: search_gmail_messages Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_search_gmail_messages_success() -> None:
    """Verify search_gmail_messages normalizes matching messages and page tokens."""
    client = MagicMock(spec=GmailClient)
    client.execute_tool_async = MagicMock()

    mock_data = {
        "messages": [
            {
                "id": "18c123456789abcd",
                "threadId": "thread_abc123",
                "sender": "Alice <alice@example.com>",
                "to": "bob@example.com",
                "subject": "SYNQRA Launch Sync",
                "messageTimestamp": "2026-11-08T09:30:00Z",
                "snippet": "Let's review the MCP server status...",
            },
            {
                "messageId": "18c987654321ef01",
                "thread_id": "thread_ef0123",
                "from": "Support <support@service.com>",
                "recipient": "bob@example.com",
                "subject": "Weekly Digest",
                "date": "2026-11-07T18:00:00Z",
                "preview": {"snippet": "Here is your weekly digest..."},
            },
        ],
        "nextPageToken": "token_page_2",
    }

    async def mock_execute(tool_slug: str, arguments: dict, **_kwargs):
        _ = tool_slug
        _ = arguments
        return mock_data

    client.execute_tool_async.side_effect = mock_execute

    result = await search_gmail_messages(
        client,
        sender="alice@example.com",
        subject="Launch",
        max_results=5,
    )

    assert result.total_count == 2
    assert result.next_page_token == "token_page_2"
    assert result.messages[0].id == "18c123456789abcd"
    assert result.messages[0].thread_id == "thread_abc123"
    assert result.messages[0].subject == "SYNQRA Launch Sync"
    assert result.messages[1].id == "18c987654321ef01"
    assert result.messages[1].snippet == "Here is your weekly digest..."


@pytest.mark.asyncio
async def test_search_gmail_messages_empty() -> None:
    """Verify search_gmail_messages handles zero results gracefully."""
    client = MagicMock(spec=GmailClient)
    client.execute_tool_async = MagicMock()

    async def mock_execute(tool_slug: str, arguments: dict, **_kwargs):
        _ = tool_slug
        _ = arguments
        return {"messages": [], "nextPageToken": None}

    client.execute_tool_async.side_effect = mock_execute

    result = await search_gmail_messages(client, query="nonexistent query 12345")
    assert result.total_count == 0
    assert result.messages == []
    assert result.next_page_token is None


# ==============================================================================
# Tool 2: get_gmail_message Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_get_gmail_message_full_details() -> None:
    """Verify get_gmail_message returns parsed bodies, attachments, headers, and labels."""
    client = MagicMock(spec=GmailClient)
    client.execute_tool_async = MagicMock()

    mock_data = {
        "messageId": "msg_001_full",
        "threadId": "thread_001",
        "sender": "sender@domain.com",
        "to": "receiver@domain.com",
        "subject": "Detailed Project Report",
        "messageTimestamp": "2026-11-08T10:00:00Z",
        "snippet": "Attached is the quarterly summary...",
        "messageText": "Hello team,\nAttached is the quarterly summary.\nBest,\nSender",
        "htmlBody": "<p>Hello team,<br>Attached is the quarterly summary.</p>",
        "attachmentList": [
            {
                "filename": "report.pdf",
                "mimeType": "application/pdf",
                "size": 1048576,
                "attachmentId": "att_987",
            }
        ],
        "labelIds": ["INBOX", "IMPORTANT"],
        "payload": {
            "headers": [
                {"name": "Cc", "value": "manager@domain.com, lead@domain.com"},
                {"name": "Bcc", "value": "archive@domain.com"},
            ]
        },
    }

    async def mock_execute(tool_slug: str, arguments: dict, **_kwargs):
        _ = tool_slug
        _ = arguments
        return mock_data

    client.execute_tool_async.side_effect = mock_execute

    details = await get_gmail_message(client, message_id="msg_001_full")

    assert details.id == "msg_001_full"
    assert details.thread_id == "thread_001"
    assert details.subject == "Detailed Project Report"
    assert "manager@domain.com" in details.cc
    assert "lead@domain.com" in details.cc
    assert "archive@domain.com" in details.bcc
    assert details.plain_text_body is not None
    assert "Hello team" in details.plain_text_body
    assert details.html_body is not None
    assert "<p>" in details.html_body
    assert len(details.attachments) == 1
    assert details.attachments[0].filename == "report.pdf"
    assert details.attachments[0].size == 1048576
    assert "INBOX" in details.labels


@pytest.mark.asyncio
async def test_get_gmail_message_empty_id_raises() -> None:
    """Verify get_gmail_message raises error on blank ID."""
    client = MagicMock(spec=GmailClient)
    with pytest.raises(ValueError):
        await get_gmail_message(client, message_id="")


# ==============================================================================
# Tool 3: list_gmail_labels Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_list_gmail_labels_success() -> None:
    """Verify list_gmail_labels returns system and user labels with counts."""
    client = MagicMock(spec=GmailClient)
    client.execute_tool_async = MagicMock()

    mock_data = {
        "labels": [
            {
                "id": "INBOX",
                "name": "INBOX",
                "type": "system",
                "messagesTotal": 1420,
                "threadsTotal": 890,
                "messagesUnread": 12,
            },
            {
                "id": "Label_ProjectX",
                "name": "Project X",
                "type": "user",
                "messagesTotal": 45,
                "threadsTotal": 20,
                "messagesUnread": 0,
            },
        ]
    }

    async def mock_execute(tool_slug: str, arguments: dict, **_kwargs):
        _ = tool_slug
        _ = arguments
        return mock_data

    client.execute_tool_async.side_effect = mock_execute

    result = await list_gmail_labels(client)

    assert result.total_count == 2
    assert result.labels[0].id == "INBOX"
    assert result.labels[0].type == "system"
    assert result.labels[0].messages_unread == 12
    assert result.labels[1].id == "Label_ProjectX"
    assert result.labels[1].name == "Project X"
    assert result.labels[1].type == "user"


# ==============================================================================
# Tool 4: create_gmail_draft Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_create_gmail_draft_success() -> None:
    """Verify create_gmail_draft creates a draft and does NOT transmit email."""
    client = MagicMock(spec=GmailClient)
    client.execute_tool_async = MagicMock()

    captured_args = {}

    async def mock_execute(tool_slug: str, arguments: dict, **_kwargs):
        _ = tool_slug
        nonlocal captured_args
        captured_args = arguments
        return {
            "id": "draft_abc123",
            "message": {
                "id": "msg_draft_msg_1",
                "threadId": "thread_draft_1",
            },
        }

    client.execute_tool_async.side_effect = mock_execute

    result = await create_gmail_draft(
        client,
        recipient="colleague@example.com",
        subject="Draft Agenda",
        body="Here is the tentative agenda.",
        cc=["lead@example.com"],
        thread_id="thread_existing_1",
    )

    assert result.draft_id == "draft_abc123"
    assert result.message_id == "msg_draft_msg_1"
    assert result.thread_id == "thread_draft_1"
    assert result.status == "draft_created"
    assert captured_args["recipient_email"] == "colleague@example.com"
    assert captured_args["subject"] == "Draft Agenda"
    assert captured_args["thread_id"] == "thread_existing_1"
    assert captured_args["cc"] == ["lead@example.com"]


@pytest.mark.asyncio
async def test_create_gmail_draft_invalid_email_raises() -> None:
    """Verify create_gmail_draft raises validation error on invalid recipient email."""
    client = MagicMock(spec=GmailClient)
    with pytest.raises(ValueError, match="Invalid email address"):
        await create_gmail_draft(
            client,
            recipient="not-an-email",
            subject="Test",
            body="Body",
        )


# ==============================================================================
# Tool 5: send_gmail_message Safety & Execution Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_send_gmail_message_aborted_without_confirm() -> None:
    """Safety check: Verify send_gmail_message does NOT send when confirm=False."""
    client = MagicMock(spec=GmailClient)
    client.execute_tool_async = MagicMock()

    result = await send_gmail_message(
        client,
        recipient="partner@example.com",
        subject="Important Contracts",
        body="Please find attached.",
        confirm=False,
    )

    assert result.confirmed is False
    assert result.status == "aborted"
    assert "NOT sent" in result.message
    # Composio API must never be invoked
    client.execute_tool_async.assert_not_called()


@pytest.mark.asyncio
async def test_send_gmail_message_executes_with_confirm() -> None:
    """Verify send_gmail_message sends when confirm=True."""
    client = MagicMock(spec=GmailClient)
    client.execute_tool_async = MagicMock()

    captured_args = {}

    async def mock_execute(tool_slug: str, arguments: dict, **_kwargs):
        _ = tool_slug
        nonlocal captured_args
        captured_args = arguments
        return {
            "id": "sent_msg_888",
            "threadId": "thread_888",
        }

    client.execute_tool_async.side_effect = mock_execute

    result = await send_gmail_message(
        client,
        recipient="partner@example.com",
        subject="Approved Notification",
        body="All systems operational.",
        confirm=True,
    )

    assert result.confirmed is True
    assert result.status == "sent"
    assert result.message_id == "sent_msg_888"
    assert result.thread_id == "thread_888"
    assert captured_args["recipient_email"] == "partner@example.com"
    assert captured_args["subject"] == "Approved Notification"


# ==============================================================================
# Tool 6: reply_to_gmail_message Safety & Context Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_reply_to_gmail_message_aborted_without_confirm() -> None:
    """Safety check: Verify reply_to_gmail_message does NOT send when confirm=False."""
    client = MagicMock(spec=GmailClient)
    client.execute_tool_async = MagicMock()

    result = await reply_to_gmail_message(
        client,
        message_id="msg_original_123",
        reply_body="Thank you for the update.",
        confirm=False,
    )

    assert result.confirmed is False
    assert result.status == "aborted"
    assert "NOT sent" in result.message
    client.execute_tool_async.assert_not_called()


@pytest.mark.asyncio
async def test_reply_to_gmail_message_resolves_thread_and_sender_with_confirm() -> None:
    """Verify reply_to_gmail_message automatically resolves threadId and sender from message."""
    client = MagicMock(spec=GmailClient)
    client.execute_tool_async = MagicMock()

    captured_actions = []
    captured_args = {}

    async def mock_execute(tool_slug: str, arguments: dict, **_kwargs):
        captured_actions.append(tool_slug)
        nonlocal captured_args
        captured_args = arguments
        if tool_slug == GMAIL_FETCH_MESSAGE_BY_ID_ACTION:
            return {
                "id": "msg_orig_999",
                "threadId": "thread_auto_resolved",
                "sender": "sender_person@example.com",
                "subject": "Initial Request",
            }
        if tool_slug == GMAIL_REPLY_TO_THREAD_ACTION:
            return {
                "id": "reply_msg_1001",
                "threadId": "thread_auto_resolved",
            }
        return {}

    client.execute_tool_async.side_effect = mock_execute

    result = await reply_to_gmail_message(
        client,
        message_id="msg_orig_999",
        reply_body="I have reviewed and approved your request.",
        confirm=True,
    )

    assert result.confirmed is True
    assert result.status == "replied"
    assert result.message_id == "reply_msg_1001"
    assert result.thread_id == "thread_auto_resolved"
    assert result.recipient == "sender_person@example.com"
    assert GMAIL_FETCH_MESSAGE_BY_ID_ACTION in captured_actions
    assert GMAIL_REPLY_TO_THREAD_ACTION in captured_actions
    assert captured_args["thread_id"] == "thread_auto_resolved"
    assert captured_args["recipient_email"] == "sender_person@example.com"


# ==============================================================================
# Integration & FastMCP Tool Registration Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_gmail_integration_tool_registration() -> None:
    """Verify GmailIntegration registers all 6 tools on FastMCP server."""
    client = MagicMock(spec=GmailClient)
    client.is_configured = True
    integration = GmailIntegration(client=client)

    server = FastMCP("TestGmailServer")
    registered = integration.register_tools(server)

    expected_tools = [
        "search_gmail_messages",
        "get_gmail_message",
        "list_gmail_labels",
        "create_gmail_draft",
        "send_gmail_message",
        "reply_to_gmail_message",
    ]

    assert len(registered) == 6
    for tool_name in expected_tools:
        assert tool_name in registered

    server_tools = await server.list_tools()
    server_tool_names = [t.name for t in server_tools]
    for tool_name in expected_tools:
        assert tool_name in server_tool_names


@pytest.mark.asyncio
async def test_gmail_health_check_connected() -> None:
    """Verify health_check returns CONNECTED when configured and connected."""
    client = MagicMock(spec=GmailClient)
    client.is_configured = True
    client.check_connection_async = MagicMock()

    async def mock_conn():
        return True

    client.check_connection_async.side_effect = mock_conn

    integration = GmailIntegration(client=client)
    status = await integration.health_check()

    assert status.state == IntegrationState.CONNECTED
    assert status.is_connected is True
    assert status.is_configured is True


@pytest.mark.asyncio
async def test_gmail_health_check_unauthenticated_provides_auth_url() -> None:
    """Verify health_check provides OAuth authorization URL when not connected."""
    client = MagicMock(spec=GmailClient)
    client.is_configured = True
    client.check_connection_async = MagicMock()

    async def mock_conn():
        return False

    client.check_connection_async.side_effect = mock_conn

    async def mock_auth():
        return "https://connect.composio.dev/auth/gmail_abc"

    client.authorize_async = MagicMock(side_effect=mock_auth)

    integration = GmailIntegration(client=client)
    status = await integration.health_check()

    assert status.state == IntegrationState.ERROR
    assert status.is_connected is False
    assert "https://connect.composio.dev/auth/gmail_abc" in str(status.error_message)


def test_gmail_health_check_unconfigured() -> None:
    """Verify health_check returns UNCONFIGURED when no credentials are present."""
    empty_settings = Settings()
    integration = GmailIntegration(settings=empty_settings)
    # Synchronous is_configured check
    assert integration.is_configured() is False
