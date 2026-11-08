"""Pydantic schemas for Gmail integration in SYNQRA.

Defines validated input parameters and structured response models for email searching,
message inspection, label discovery, draft creation, email sending, and replying.
"""

import re

from pydantic import BaseModel, Field, field_validator

EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def validate_email_address(v: str) -> str:
    """Validate a single email address."""
    clean = v.strip()
    if not clean or not EMAIL_REGEX.match(clean):
        raise ValueError(f"Invalid email address: '{v}'. Must be in format user@example.com.")
    return clean


class GmailAttachmentMetadata(BaseModel):
    """Metadata describing a file attachment on an email message."""

    filename: str = Field(description="Name of the attached file.")
    mime_type: str | None = Field(default=None, description="MIME type of the attachment.")
    size: int | None = Field(default=None, description="Size of the attachment in bytes.")
    attachment_id: str | None = Field(
        default=None, description="Technical ID for fetching raw attachment bytes."
    )


class GmailMessageSummary(BaseModel):
    """Normalized summary of a Gmail email message."""

    id: str = Field(description="Unique message ID.")
    thread_id: str = Field(description="Thread identifier containing this message.")
    sender: str | None = Field(default=None, description="Sender email address or display name.")
    recipient: str | None = Field(default=None, description="Primary recipient email address.")
    subject: str | None = Field(default=None, description="Subject line of the email.")
    date: str | None = Field(default=None, description="Date and time the message was sent.")
    snippet: str | None = Field(default=None, description="Brief preview snippet of email content.")


class SearchGmailMessagesInput(BaseModel):
    """Input parameters for searching messages in Gmail."""

    query: str | None = Field(
        default=None,
        description="Gmail advanced search query string (e.g. 'is:unread', 'has:attachment').",
    )
    sender: str | None = Field(
        default=None,
        description="Filter messages sent by this email address or name (appends 'from:...').",
    )
    recipient: str | None = Field(
        default=None,
        description="Filter messages received by this address (appends 'to:...').",
    )
    subject: str | None = Field(
        default=None,
        description="Filter messages with this text in the subject (appends 'subject:...').",
    )
    keywords: str | None = Field(
        default=None,
        description="Search terms to match in email body or metadata.",
    )
    after: str | None = Field(
        default=None,
        description="Filter messages sent on or after this date (YYYY/MM/DD or YYYY-MM-DD).",
    )
    before: str | None = Field(
        default=None,
        description="Filter messages sent before this date (YYYY/MM/DD or YYYY-MM-DD).",
    )
    label_ids: list[str] | None = Field(
        default=None,
        description="Filter messages having these internal label IDs (e.g. ['INBOX', 'STARRED']).",
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of messages to return (1-100).",
    )
    page_token: str | None = Field(
        default=None,
        description="Token for retrieving the next page of results.",
    )
    include_spam_trash: bool = Field(
        default=False,
        description="Whether to include messages from SPAM and TRASH.",
    )


class SearchGmailMessagesOutput(BaseModel):
    """Structured response containing Gmail search results."""

    total_count: int = Field(description="Number of messages returned in this page.")
    messages: list[GmailMessageSummary] = Field(
        default_factory=list, description="Matching messages."
    )
    next_page_token: str | None = Field(
        default=None, description="Pagination token for the next page of messages."
    )


class GetGmailMessageInput(BaseModel):
    """Input parameters for retrieving an individual Gmail message."""

    message_id: str = Field(description="Unique Gmail message ID to retrieve.", min_length=1)
    format: str = Field(
        default="full",
        description="Representation format ('full', 'metadata', 'minimal').",
    )


class GmailMessageDetails(BaseModel):
    """Detailed contents and headers of a Gmail message."""

    id: str = Field(description="Unique message ID.")
    thread_id: str = Field(description="Thread identifier containing this message.")
    sender: str | None = Field(default=None, description="Sender email address or name.")
    recipient: str | None = Field(default=None, description="Primary recipient.")
    cc: list[str] = Field(default_factory=list, description="Carbon copy (CC) recipients.")
    bcc: list[str] = Field(default_factory=list, description="Blind carbon copy (BCC) recipients.")
    subject: str | None = Field(default=None, description="Subject line.")
    date: str | None = Field(default=None, description="Date header or internal timestamp.")
    snippet: str | None = Field(default=None, description="Short snippet preview.")
    plain_text_body: str | None = Field(default=None, description="Extracted plain text body.")
    html_body: str | None = Field(default=None, description="Extracted HTML body when available.")
    attachments: list[GmailAttachmentMetadata] = Field(
        default_factory=list, description="List of file attachments."
    )
    labels: list[str] = Field(default_factory=list, description="Label IDs applied to the message.")


class GmailLabelSummary(BaseModel):
    """Normalized summary of a Gmail label."""

    id: str = Field(description="Internal label ID (e.g. 'INBOX', 'Label_123').")
    name: str = Field(description="Display name of the label.")
    type: str = Field(description="Label type ('system' or 'user').")
    messages_total: int | None = Field(default=None, description="Total messages with this label.")
    threads_total: int | None = Field(default=None, description="Total threads with this label.")
    messages_unread: int | None = Field(
        default=None, description="Number of unread messages with this label."
    )


class ListGmailLabelsInput(BaseModel):
    """Input parameters for listing Gmail labels."""

    include_details: bool = Field(
        default=True,
        description="Whether to include message counts and thread statistics.",
    )


class ListGmailLabelsOutput(BaseModel):
    """Structured response containing Gmail mailbox labels."""

    total_count: int = Field(description="Number of labels retrieved.")
    labels: list[GmailLabelSummary] = Field(
        default_factory=list, description="List of accessible mailbox labels."
    )


class CreateGmailDraftInput(BaseModel):
    """Input parameters for creating an email draft without sending."""

    recipient: str = Field(description="Primary recipient email address.", min_length=3)
    subject: str = Field(description="Subject line for the draft email.")
    body: str = Field(description="Body content for the draft email.")
    cc: list[str] | None = Field(default=None, description="Optional CC recipient emails.")
    bcc: list[str] | None = Field(default=None, description="Optional BCC recipient emails.")
    is_html: bool = Field(default=False, description="Set to True if body contains HTML tags.")
    thread_id: str | None = Field(
        default=None, description="Thread ID to attach the draft to an existing thread."
    )

    @field_validator("recipient")
    @classmethod
    def check_recipient(cls, v: str) -> str:
        return validate_email_address(v)


class CreateGmailDraftOutput(BaseModel):
    """Structured response for email draft creation."""

    draft_id: str = Field(description="Identifier of the newly created draft.")
    message_id: str | None = Field(
        default=None, description="Message ID associated with the draft."
    )
    thread_id: str | None = Field(default=None, description="Thread ID containing the draft.")
    recipient: str = Field(description="Recipient email.")
    subject: str = Field(description="Draft subject.")
    status: str = Field(default="draft_created", description="Draft creation status.")
    message: str = Field(description="Human-readable confirmation message.")


class SendGmailMessageInput(BaseModel):
    """Input parameters for sending an email with mandatory confirmation safety check."""

    recipient: str = Field(description="Primary recipient email address.", min_length=3)
    subject: str = Field(description="Subject line of the email.")
    body: str = Field(description="Email body text or HTML.")
    cc: list[str] | None = Field(default=None, description="Optional CC recipient emails.")
    bcc: list[str] | None = Field(default=None, description="Optional BCC recipient emails.")
    is_html: bool = Field(
        default=False, description="Set to True if body contains HTML formatting."
    )
    confirm: bool = Field(
        default=False,
        description="Mandatory confirmation flag. Must be explicitly set to True to send email.",
    )

    @field_validator("recipient")
    @classmethod
    def check_recipient(cls, v: str) -> str:
        return validate_email_address(v)


class SendGmailMessageOutput(BaseModel):
    """Structured response for email transmission."""

    message_id: str | None = Field(default=None, description="ID of the sent message.")
    thread_id: str | None = Field(default=None, description="Thread ID of the sent message.")
    status: str = Field(description="Transmission status ('sent', 'aborted', 'failed').")
    confirmed: bool = Field(description="Whether explicit confirmation was provided.")
    recipient: str = Field(description="Target recipient.")
    subject: str = Field(description="Subject line.")
    message: str = Field(description="Human-readable status summary.")


class ReplyToGmailMessageInput(BaseModel):
    """Input parameters for replying to an existing email with mandatory confirmation."""

    message_id: str = Field(
        description="Message ID of the email to reply to.",
        min_length=1,
    )
    reply_body: str = Field(description="Body content for the reply message.", min_length=1)
    thread_id: str | None = Field(
        default=None,
        description="Thread ID to reply within. If omitted, looked up from message_id.",
    )
    recipient: str | None = Field(
        default=None,
        description="Recipient email for reply. If omitted, looked up from original sender.",
    )
    cc: list[str] | None = Field(default=None, description="Optional CC recipient emails.")
    bcc: list[str] | None = Field(default=None, description="Optional BCC recipient emails.")
    is_html: bool = Field(default=False, description="Set to True if reply contains HTML.")
    confirm: bool = Field(
        default=False,
        description="Mandatory confirmation flag. Must be explicitly set to True to send reply.",
    )


class ReplyToGmailMessageOutput(BaseModel):
    """Structured response for reply transmission."""

    message_id: str | None = Field(
        default=None, description="ID of the newly created reply message."
    )
    thread_id: str | None = Field(default=None, description="Thread ID containing the reply.")
    status: str = Field(description="Reply status ('replied', 'aborted', 'failed').")
    confirmed: bool = Field(description="Whether explicit confirmation was provided.")
    recipient: str | None = Field(default=None, description="Recipient of the reply.")
    message: str = Field(description="Human-readable status summary.")
