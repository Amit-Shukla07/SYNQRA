"""Cryptographic verification, tamper detection, and security binding for approvals.

Provides canonical argument serialization, SHA-256 fingerprinting, HMAC signature
generation, single-use binding, and strict human identity validation.
"""

import hashlib
import hmac
import json
from datetime import datetime
from typing import Any

from synqra.agents.approval.models import ApprovalRequestRecord
from synqra.agents.schemas import (
    DISALLOWED_DECIDED_BY_IDENTIFIERS,
    ApprovalStatus,
    PermissionLevel,
)

DEFAULT_APPROVAL_SECRET: str = "synqra-insecure-dev-approval-secret-key-change-in-production"


def canonical_json_serialize(data: Any) -> str:
    """Serialize arbitrary data structures into deterministic, canonical JSON format.

    Sorts keys recursively, handles nested dictionaries and lists, and formats primitives
    without non-deterministic whitespace.
    """

    def _normalize(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _normalize(v) for k, v in sorted(obj.items())}
        if isinstance(obj, (list, tuple)):
            return [_normalize(item) for item in obj]
        if isinstance(obj, datetime):
            return obj.isoformat()
        return obj

    normalized = _normalize(data)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_arguments_hash(arguments: dict[str, Any]) -> str:
    """Generate a cryptographic SHA-256 hex digest of canonically serialized arguments."""
    canonical = canonical_json_serialize(arguments)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def generate_approval_binding_token(
    approval_id: str,
    workflow_id: str,
    thread_id: str,
    step_id: str,
    tool_name: str,
    arguments_hash: str,
    permission_level: str | PermissionLevel,
    request_version: int,
    secret_key: str = DEFAULT_APPROVAL_SECRET,
) -> str:
    """Generate a verifiable HMAC-SHA256 signature binding the approval to execution context."""
    perm_val = (
        permission_level.value
        if isinstance(permission_level, PermissionLevel)
        else str(permission_level)
    )
    payload = (
        f"{approval_id}|{workflow_id}|{thread_id}|{step_id}|"
        f"{tool_name}|{arguments_hash}|{perm_val}|{request_version}"
    )
    return hmac.new(
        secret_key.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def validate_human_actor(actor: str | None) -> tuple[bool, str | None]:
    """Verify that the decision maker is an authenticated human operator, not an automated LLM.

    Returns:
        (is_valid, rejection_reason)
    """
    if not actor or not actor.strip():
        return False, "Approval actor cannot be empty or unspecified."

    clean_actor = actor.strip().lower()
    if clean_actor in DISALLOWED_DECIDED_BY_IDENTIFIERS:
        return (
            False,
            f"Disallowed decider '{actor}': Automated models, LLMs, or autonomous agents "
            f"cannot issue approval decisions.",
        )

    return True, None


def verify_approval_binding(
    record: ApprovalRequestRecord,
    workflow_id: str,
    thread_id: str,
    step_id: str,
    tool_name: str,
    candidate_arguments: dict[str, Any],
    candidate_permission_level: PermissionLevel | None = None,
    secret_key: str = DEFAULT_APPROVAL_SECRET,
    at_time: datetime | None = None,
) -> tuple[bool, str | None]:
    """Comprehensively verify that an approval record satisfies strict execution binding.

    Checks:
    1. Workflow ID matching
    2. Thread ID matching
    3. Step ID matching
    4. Target tool name matching
    5. Arguments hash matching (arguments unchanged since approval)
    6. Permission tier matching
    7. Approval lifecycle status is APPROVED
    8. Expiration deadline not passed
    9. Single-use consumption status (not already consumed)
    10. Cryptographic binding token signature (HMAC)

    Returns:
        (is_valid, failure_reason)
    """
    # 1. Lifecycle status
    if record.status == ApprovalStatus.CONSUMED:
        return False, f"Approval '{record.approval_id}' has already been consumed (single-use)."
    if record.status == ApprovalStatus.REJECTED:
        return False, f"Approval '{record.approval_id}' was rejected by human operator."
    if record.status == ApprovalStatus.EXPIRED or record.is_expired(at_time=at_time):
        return False, f"Approval '{record.approval_id}' has expired."
    if record.status != ApprovalStatus.APPROVED:
        return (
            False,
            f"Approval '{record.approval_id}' is in status '{record.status.value}', not approved.",
        )

    # 2. Context binding
    if record.workflow_id != workflow_id:
        return (
            False,
            f"Workflow mismatch: approval belongs to workflow '{record.workflow_id}', "
            f"attempted by '{workflow_id}'.",
        )
    if record.thread_id != thread_id:
        return (
            False,
            f"Thread mismatch: approval belongs to thread '{record.thread_id}', "
            f"attempted by '{thread_id}'.",
        )
    if record.step_id != step_id:
        return (
            False,
            f"Step mismatch: approval was granted for step '{record.step_id}', "
            f"attempted for step '{step_id}'.",
        )
    if record.tool_name != tool_name:
        return (
            False,
            f"Tool mismatch: approval was granted for tool '{record.tool_name}', "
            f"attempted for tool '{tool_name}'.",
        )
    if candidate_permission_level and record.permission_level != candidate_permission_level:
        return (
            False,
            (
                f"Permission level mismatch: approval was granted for "
                f"'{record.permission_level.value}', attempted for "
                f"'{candidate_permission_level.value}'."
            ),
        )

    # 3. Arguments hash verification (anti-tamper)
    candidate_hash = compute_arguments_hash(candidate_arguments)
    if not hmac.compare_digest(record.arguments_hash, candidate_hash):
        return (
            False,
            f"Argument tamper detected: arguments have been modified since approval was granted. "
            f"Approved hash: {record.arguments_hash}, Execution hash: {candidate_hash}.",
        )

    # 4. HMAC Token verification
    if record.binding_token:
        expected_token = generate_approval_binding_token(
            approval_id=record.approval_id,
            workflow_id=record.workflow_id,
            thread_id=record.thread_id,
            step_id=record.step_id,
            tool_name=record.tool_name,
            arguments_hash=record.arguments_hash,
            permission_level=record.permission_level,
            request_version=record.request_version,
            secret_key=secret_key,
        )
        if not hmac.compare_digest(record.binding_token, expected_token):
            return (
                False,
                f"Forged or invalid approval binding token on request '{record.approval_id}'.",
            )

    return True, None


def format_human_readable_operation(
    tool_name: str,
    arguments: dict[str, Any],
    fallback_description: str | None = None,
) -> str:
    """Format a clear, explicit summary showing the exact operation to be executed."""
    if tool_name == "send_gmail_message":
        recipient = arguments.get("recipient", "")
        subject = arguments.get("subject", "(no subject)")
        body = arguments.get("body", "")
        body_prev = (body[:100] + "...") if len(body) > 100 else body
        return (
            f"Send email via Gmail to '{recipient}' with subject '{subject}'. "
            f"Body preview: {body_prev}"
        )

    if tool_name == "reply_to_gmail_message":
        msg_id = arguments.get("message_id", "")
        body = arguments.get("body", "")
        body_prev = (body[:100] + "...") if len(body) > 100 else body
        return f"Send reply via Gmail to message '{msg_id}'. Reply body preview: {body_prev}"

    if tool_name == "create_gmail_draft":
        recipient = arguments.get("recipient", "")
        subject = arguments.get("subject", "")
        return f"Create Gmail draft to '{recipient}' with subject '{subject}'."

    if tool_name == "delete_calendar_event":
        cal_id = arguments.get("calendar_id", "primary")
        event_id = arguments.get("event_id", "")
        return f"Permanently DELETE calendar event '{event_id}' from calendar '{cal_id}'."

    if tool_name == "create_calendar_event":
        summary = arguments.get("summary", "")
        start = arguments.get("start_time", "")
        end = arguments.get("end_time", "")
        return f"Create Google Calendar event '{summary}' scheduled from {start} to {end}."

    if tool_name == "update_calendar_event":
        cal_id = arguments.get("calendar_id", "primary")
        event_id = arguments.get("event_id", "")
        return f"Update Google Calendar event '{event_id}' in calendar '{cal_id}'."

    if tool_name == "create_google_document":
        title = arguments.get("title", "")
        return f"Create Google Drive document titled '{title}'."

    if tool_name == "update_google_document":
        doc_id = arguments.get("document_id", "")
        return f"Update content in Google Drive document '{doc_id}'."

    if tool_name == "upload_drive_file":
        name = arguments.get("name", arguments.get("file_name", "unnamed"))
        return f"Upload file '{name}' to Google Drive."

    if tool_name == "create_drive_folder":
        name = arguments.get("name", "")
        return f"Create folder '{name}' on Google Drive."

    return fallback_description or f"Execute {tool_name} with arguments: {arguments}"
