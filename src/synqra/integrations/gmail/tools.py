"""Gmail tool implementations for SYNQRA.

Provides core business logic for email searching, message inspection, label retrieval,
draft creation, email sending with confirmation, and thread replying via Composio.
"""

import logging
from typing import Any

from synqra.integrations.exceptions import IntegrationExecutionError
from synqra.integrations.gmail.client import (
    GMAIL_CREATE_DRAFT_ACTION,
    GMAIL_FETCH_EMAILS_ACTION,
    GMAIL_FETCH_MESSAGE_BY_ID_ACTION,
    GMAIL_LIST_LABELS_ACTION,
    GMAIL_REPLY_TO_THREAD_ACTION,
    GMAIL_SEND_EMAIL_ACTION,
    GmailClient,
)
from synqra.integrations.gmail.schemas import (
    CreateGmailDraftOutput,
    GetGmailMessageInput,
    GmailAttachmentMetadata,
    GmailLabelSummary,
    GmailMessageDetails,
    GmailMessageSummary,
    ListGmailLabelsOutput,
    ReplyToGmailMessageOutput,
    SearchGmailMessagesOutput,
    SendGmailMessageOutput,
    validate_email_address,
)

logger = logging.getLogger("synqra.integrations.gmail.tools")


def build_gmail_query(
    query: str | None = None,
    sender: str | None = None,
    recipient: str | None = None,
    subject: str | None = None,
    keywords: str | None = None,
    after: str | None = None,
    before: str | None = None,
) -> str | None:
    """Build a composite Gmail search query from individual filter parameters.

    Args:
        query: Base or raw Gmail search query.
        sender: Sender filter (from:...).
        recipient: Recipient filter (to:...).
        subject: Subject filter (subject:...).
        keywords: Free-text keywords.
        after: Lower date boundary (after:...).
        before: Upper date boundary (before:...).

    Returns:
        Combined search query string or None if no filters provided.
    """
    parts: list[str] = []
    if sender:
        parts.append(f"from:{sender.strip()}")
    if recipient:
        parts.append(f"to:{recipient.strip()}")
    if subject:
        parts.append(f'subject:"{subject.strip()}"')
    if keywords:
        parts.append(keywords.strip())
    if after:
        parts.append(f"after:{after.strip()}")
    if before:
        parts.append(f"before:{before.strip()}")
    if query:
        parts.append(query.strip())

    combined = " ".join(parts).strip()
    return combined if combined else None


async def search_gmail_messages(
    client: GmailClient,
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
    """Search Gmail messages using text queries, field filters, and label criteria.

    Args:
        client: Active GmailClient instance.
        query: Raw advanced search string.
        sender: Filter by sender.
        recipient: Filter by recipient.
        subject: Filter by subject line.
        keywords: Free text keyword search.
        after: Date filter (on or after).
        before: Date filter (before).
        label_ids: List of label IDs to require.
        max_results: Maximum messages to retrieve per page.
        page_token: Optional token for retrieving next page.
        include_spam_trash: Whether to include SPAM and TRASH folders.

    Returns:
        SearchGmailMessagesOutput with normalized message previews.
    """
    args: dict[str, Any] = {
        "max_results": max_results,
        "include_spam_trash": include_spam_trash,
    }

    full_query = build_gmail_query(
        query=query,
        sender=sender,
        recipient=recipient,
        subject=subject,
        keywords=keywords,
        after=after,
        before=before,
    )
    if full_query:
        args["query"] = full_query
    if label_ids:
        args["label_ids"] = label_ids
    if page_token:
        args["page_token"] = page_token

    data = await client.execute_tool_async(GMAIL_FETCH_EMAILS_ACTION, args)

    raw_messages = data.get("messages") or []
    next_page_token = data.get("nextPageToken") or data.get("next_page_token")

    normalized: list[GmailMessageSummary] = []
    for m in raw_messages:
        if not isinstance(m, dict):
            continue
        msg_id = m.get("messageId") or m.get("id") or ""
        thread_id = m.get("threadId") or m.get("thread_id") or ""
        msg_sender = m.get("sender") or m.get("from")
        msg_to = m.get("to") or m.get("recipient")
        msg_subject = m.get("subject")
        msg_date = m.get("messageTimestamp") or m.get("date") or m.get("internalDate")
        snippet = m.get("snippet") or (m.get("preview") or {}).get("snippet")

        normalized.append(
            GmailMessageSummary(
                id=msg_id,
                thread_id=thread_id,
                sender=msg_sender,
                recipient=msg_to,
                subject=msg_subject,
                date=str(msg_date) if msg_date is not None else None,
                snippet=snippet,
            )
        )

    return SearchGmailMessagesOutput(
        total_count=len(normalized),
        messages=normalized,
        next_page_token=next_page_token,
    )


async def get_gmail_message(
    client: GmailClient,
    message_id: str,
    format: str = "full",
) -> GmailMessageDetails:
    """Retrieve full details of an individual email message.

    Args:
        client: Active GmailClient instance.
        message_id: Unique Gmail message ID.
        format: Format representation ('full', 'metadata', 'minimal').

    Returns:
        GmailMessageDetails with headers, bodies, and attachment metadata.
    """
    valid_id = GetGmailMessageInput(message_id=message_id, format=format).message_id

    args = {
        "message_id": valid_id,
        "format": format,
    }

    data = await client.execute_tool_async(GMAIL_FETCH_MESSAGE_BY_ID_ACTION, args)
    item = data.get("message") if isinstance(data.get("message"), dict) else data

    msg_id = item.get("messageId") or item.get("id") or valid_id
    thread_id = item.get("threadId") or item.get("thread_id") or ""
    sender = item.get("sender") or item.get("from")
    recipient = item.get("to") or item.get("recipient")
    subject = item.get("subject")
    date_val = item.get("messageTimestamp") or item.get("date")
    snippet = item.get("snippet") or (item.get("preview") or {}).get("snippet")
    plain_text = item.get("messageText")
    html_body = item.get("htmlBody") or item.get("html")

    # Extract headers if present in payload
    payload = item.get("payload") or {}
    headers = payload.get("headers") or []
    cc_list: list[str] = []
    bcc_list: list[str] = []

    if isinstance(headers, list):
        for h in headers:
            if isinstance(h, dict):
                h_name = (h.get("name") or "").lower()
                h_val = h.get("value") or ""
                if h_name == "cc" and h_val:
                    cc_list.extend([addr.strip() for addr in h_val.split(",") if addr.strip()])
                elif h_name == "bcc" and h_val:
                    bcc_list.extend([addr.strip() for addr in h_val.split(",") if addr.strip()])
                elif h_name == "subject" and not subject:
                    subject = h_val
                elif h_name == "from" and not sender:
                    sender = h_val
                elif h_name == "to" and not recipient:
                    recipient = h_val
                elif h_name == "date" and not date_val:
                    date_val = h_val

    # Parse attachments
    raw_attachments = item.get("attachmentList") or item.get("attachments") or []
    attachments: list[GmailAttachmentMetadata] = []
    for att in raw_attachments:
        if isinstance(att, dict):
            attachments.append(
                GmailAttachmentMetadata(
                    filename=att.get("filename") or att.get("name") or "attachment",
                    mime_type=att.get("mimeType") or att.get("type"),
                    size=att.get("size") or att.get("fileSize"),
                    attachment_id=att.get("attachmentId") or att.get("id"),
                )
            )

    labels = item.get("labelIds") or item.get("labels") or []

    return GmailMessageDetails(
        id=msg_id,
        thread_id=thread_id,
        sender=sender,
        recipient=recipient,
        cc=cc_list,
        bcc=bcc_list,
        subject=subject,
        date=str(date_val) if date_val is not None else None,
        snippet=snippet,
        plain_text_body=plain_text,
        html_body=html_body,
        attachments=attachments,
        labels=labels if isinstance(labels, list) else [],
    )


async def list_gmail_labels(
    client: GmailClient,
    include_details: bool = True,
) -> ListGmailLabelsOutput:
    """Retrieve all system and user labels in the user's Gmail account.

    Args:
        client: Active GmailClient instance.
        include_details: Whether to fetch counts and unread statistics.

    Returns:
        ListGmailLabelsOutput containing normalized label summaries.
    """
    args = {"include_details": include_details}
    data = await client.execute_tool_async(GMAIL_LIST_LABELS_ACTION, args)

    raw_labels = data.get("labels") or []
    labels: list[GmailLabelSummary] = []

    for lbl in raw_labels:
        if not isinstance(lbl, dict):
            continue
        labels.append(
            GmailLabelSummary(
                id=lbl.get("id") or "",
                name=lbl.get("name") or "Unnamed Label",
                type=lbl.get("type") or "user",
                messages_total=lbl.get("messagesTotal"),
                threads_total=lbl.get("threadsTotal"),
                messages_unread=lbl.get("messagesUnread"),
            )
        )

    return ListGmailLabelsOutput(
        total_count=len(labels),
        labels=labels,
    )


async def create_gmail_draft(
    client: GmailClient,
    recipient: str,
    subject: str,
    body: str,
    cc: list[str] | None = None,
    bcc: list[str] | None = None,
    is_html: bool = False,
    thread_id: str | None = None,
) -> CreateGmailDraftOutput:
    """Create a Gmail email draft without transmitting it.

    Args:
        client: Active GmailClient instance.
        recipient: Primary recipient email address.
        subject: Subject line.
        body: Plain-text or HTML content.
        cc: Optional CC email addresses.
        bcc: Optional BCC email addresses.
        is_html: True if body contains HTML tags.
        thread_id: Optional existing thread to associate the draft with.

    Returns:
        CreateGmailDraftOutput with draft and thread identifiers.
    """
    valid_recipient = validate_email_address(recipient)

    args: dict[str, Any] = {
        "recipient_email": valid_recipient,
        "subject": subject,
        "body": body,
        "is_html": is_html,
    }

    if cc:
        args["cc"] = [validate_email_address(c) for c in cc if c.strip()]
    if bcc:
        args["bcc"] = [validate_email_address(b) for b in bcc if b.strip()]
    if thread_id:
        args["thread_id"] = thread_id.strip()

    data = await client.execute_tool_async(GMAIL_CREATE_DRAFT_ACTION, args)
    msg_obj = data.get("message") or {}

    draft_id = data.get("id") or data.get("draft_id") or "draft_created"
    msg_id = msg_obj.get("id")
    resolved_thread_id = msg_obj.get("threadId") or thread_id

    return CreateGmailDraftOutput(
        draft_id=draft_id,
        message_id=msg_id,
        thread_id=resolved_thread_id,
        recipient=valid_recipient,
        subject=subject,
        status="draft_created",
        message=f"Draft successfully created for '{valid_recipient}' with subject '{subject}'.",
    )


async def send_gmail_message(
    client: GmailClient,
    recipient: str,
    subject: str,
    body: str,
    cc: list[str] | None = None,
    bcc: list[str] | None = None,
    is_html: bool = False,
    confirm: bool = False,
) -> SendGmailMessageOutput:
    """Send an email immediately with mandatory confirmation protection.

    Args:
        client: Active GmailClient instance.
        recipient: Primary recipient email address.
        subject: Subject line.
        body: Message body text or HTML.
        cc: Optional CC email addresses.
        bcc: Optional BCC email addresses.
        is_html: True if body contains HTML formatting.
        confirm: Mandatory confirmation flag. Must be True to send email.

    Returns:
        SendGmailMessageOutput with status and transmission details.
    """
    valid_recipient = validate_email_address(recipient)

    if not confirm:
        logger.warning(
            "Send request to '%s' aborted: explicit confirmation not provided.",
            valid_recipient,
        )
        return SendGmailMessageOutput(
            status="aborted",
            confirmed=False,
            recipient=valid_recipient,
            subject=subject,
            message=(
                f"Email to '{valid_recipient}' was NOT sent. Sending emails requires explicit "
                "confirmation. To confirm sending, invoke send_gmail_message with confirm=True."
            ),
        )

    args: dict[str, Any] = {
        "recipient_email": valid_recipient,
        "subject": subject,
        "body": body,
        "is_html": is_html,
    }

    if cc:
        args["cc"] = [validate_email_address(c) for c in cc if c.strip()]
    if bcc:
        args["bcc"] = [validate_email_address(b) for b in bcc if b.strip()]

    data = await client.execute_tool_async(GMAIL_SEND_EMAIL_ACTION, args)
    msg_id = data.get("id") or data.get("message_id")
    thread_id = data.get("threadId") or data.get("thread_id")

    return SendGmailMessageOutput(
        message_id=msg_id,
        thread_id=thread_id,
        status="sent",
        confirmed=True,
        recipient=valid_recipient,
        subject=subject,
        message=f"Email successfully sent to '{valid_recipient}'.",
    )


async def reply_to_gmail_message(
    client: GmailClient,
    message_id: str,
    reply_body: str,
    thread_id: str | None = None,
    recipient: str | None = None,
    cc: list[str] | None = None,
    bcc: list[str] | None = None,
    is_html: bool = False,
    confirm: bool = False,
) -> ReplyToGmailMessageOutput:
    """Reply to an existing Gmail message preserving thread context.

    Args:
        client: Active GmailClient instance.
        message_id: Original message ID being replied to.
        reply_body: Reply text or HTML.
        thread_id: Optional thread ID (looked up automatically if not provided).
        recipient: Optional recipient email (looked up automatically if not provided).
        cc: Optional CC emails.
        bcc: Optional BCC emails.
        is_html: True if reply body contains HTML.
        confirm: Mandatory confirmation flag. Must be True to send reply.

    Returns:
        ReplyToGmailMessageOutput with confirmation and status details.
    """
    if not message_id or not message_id.strip():
        raise IntegrationExecutionError(
            "message_id must not be empty.",
            integration_name="gmail",
        )
    if not reply_body or not reply_body.strip():
        raise IntegrationExecutionError(
            "reply_body must not be empty.",
            integration_name="gmail",
        )

    clean_msg_id = message_id.strip()

    if not confirm:
        logger.warning(
            "Reply request to message '%s' aborted: explicit confirmation not provided.",
            clean_msg_id,
        )
        return ReplyToGmailMessageOutput(
            status="aborted",
            confirmed=False,
            recipient=recipient,
            message=(
                f"Reply to message '{clean_msg_id}' was NOT sent. Sending replies requires "
                "confirmation. To confirm, invoke reply_to_gmail_message with confirm=True."
            ),
        )

    # Resolve thread_id and recipient from original message if not provided
    resolved_thread_id = thread_id
    resolved_recipient = recipient

    if not resolved_thread_id or not resolved_recipient:
        orig = await get_gmail_message(client, clean_msg_id)
        if not resolved_thread_id:
            resolved_thread_id = orig.thread_id
        if not resolved_recipient:
            resolved_recipient = orig.sender

    if not resolved_thread_id:
        raise IntegrationExecutionError(
            f"Could not resolve thread_id for message '{clean_msg_id}'.",
            integration_name="gmail",
        )
    if not resolved_recipient:
        raise IntegrationExecutionError(
            f"Could not resolve recipient for reply to message '{clean_msg_id}'.",
            integration_name="gmail",
        )

    valid_recipient = validate_email_address(resolved_recipient)

    args: dict[str, Any] = {
        "thread_id": resolved_thread_id.strip(),
        "recipient_email": valid_recipient,
        "message_body": reply_body,
        "is_html": is_html,
    }

    if cc:
        args["cc"] = [validate_email_address(c) for c in cc if c.strip()]
    if bcc:
        args["bcc"] = [validate_email_address(b) for b in bcc if b.strip()]

    data = await client.execute_tool_async(GMAIL_REPLY_TO_THREAD_ACTION, args)
    new_msg_id = data.get("id") or data.get("message_id")
    resp_thread_id = data.get("threadId") or data.get("thread_id") or resolved_thread_id

    return ReplyToGmailMessageOutput(
        message_id=new_msg_id,
        thread_id=resp_thread_id,
        status="replied",
        confirmed=True,
        recipient=valid_recipient,
        message=f"Reply successfully sent to '{valid_recipient}' in thread '{resolved_thread_id}'.",
    )
