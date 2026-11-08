"""Business service orchestration for Human-in-the-Loop (HITL) approvals.

Coordinates approval request creation, pending inspection, atomic operator decisions,
expiration management, and single-use atomic consumption for tool execution.
"""

import logging
from datetime import timedelta
from typing import Any

from synqra.agents.approval.models import (
    ApprovalDecisionRecord,
    ApprovalRequestRecord,
    utc_now,
)
from synqra.agents.approval.security import (
    DEFAULT_APPROVAL_SECRET,
    compute_arguments_hash,
    format_human_readable_operation,
    generate_approval_binding_token,
    validate_human_actor,
    verify_approval_binding,
)
from synqra.agents.approval.store import BaseApprovalStore, InMemoryApprovalStore
from synqra.agents.exceptions import ApprovalError
from synqra.agents.permissions import TOOL_PERMISSIONS
from synqra.agents.schemas import ApprovalStatus, PermissionLevel

logger = logging.getLogger("synqra.agents.approval.service")

# Tool to Integration name lookup
TOOL_INTEGRATION_MAP: dict[str, str] = {
    "search_youtube_videos": "youtube",
    "get_video_details": "youtube",
    "search_github_repositories": "github",
    "get_github_repository": "github",
    "get_github_file_contents": "github",
    "list_github_issues": "github",
    "get_github_readme": "github",
    "search_drive_files": "google_drive",
    "get_drive_file": "google_drive",
    "download_drive_file": "google_drive",
    "upload_drive_file": "google_drive",
    "create_google_document": "google_drive",
    "update_google_document": "google_drive",
    "create_drive_folder": "google_drive",
    "list_calendars": "google_calendar",
    "list_calendar_events": "google_calendar",
    "get_calendar_event": "google_calendar",
    "find_free_time": "google_calendar",
    "create_calendar_event": "google_calendar",
    "update_calendar_event": "google_calendar",
    "delete_calendar_event": "google_calendar",
    "search_gmail_messages": "gmail",
    "get_gmail_message": "gmail",
    "list_gmail_labels": "gmail",
    "create_gmail_draft": "gmail",
    "send_gmail_message": "gmail",
    "reply_to_gmail_message": "gmail",
    "search_web": "web_search",
    "extract_webpage_content": "web_search",
    "search_news": "web_search",
}


class ApprovalService:
    """Manages the full lifecycle of human authorization requests with concurrency safety."""

    def __init__(
        self,
        store: BaseApprovalStore | None = None,
        secret_key: str = DEFAULT_APPROVAL_SECRET,
    ) -> None:
        if store is not None:
            self.store = store
        else:
            from synqra.core.config import get_settings

            cfg = get_settings()
            if getattr(cfg, "approval_store_type", "memory") == "database":
                from synqra.agents.approval.sql_store import SqlAlchemyApprovalStore

                self.store = SqlAlchemyApprovalStore()
            else:
                self.store = InMemoryApprovalStore()
        self.secret_key = secret_key

    async def create_approval_request(
        self,
        workflow_id: str,
        thread_id: str,
        step_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        permission_level: PermissionLevel | None = None,
        description: str | None = None,
        validity_seconds: int = 900,
    ) -> ApprovalRequestRecord:
        """Create and persist a new pending approval request with cryptographic argument hash."""
        perm = permission_level or TOOL_PERMISSIONS.get(tool_name, PermissionLevel.SENSITIVE)
        integration = TOOL_INTEGRATION_MAP.get(tool_name, "general")
        arg_hash = compute_arguments_hash(arguments)
        op_description = format_human_readable_operation(
            tool_name=tool_name,
            arguments=arguments,
            fallback_description=description,
        )

        now = utc_now()
        expires = now + timedelta(seconds=max(30, validity_seconds))

        record = ApprovalRequestRecord(
            workflow_id=workflow_id,
            thread_id=thread_id,
            step_id=step_id,
            tool_name=tool_name,
            integration_name=integration,
            description=op_description,
            arguments=arguments,
            arguments_hash=arg_hash,
            permission_level=perm,
            created_at=now,
            expires_at=expires,
            status=ApprovalStatus.PENDING,
            request_version=1,
        )

        await self.store.save(record)
        logger.info(
            "Created approval request '%s' for tool '%s' (step: %s, workflow: %s)",
            record.approval_id,
            tool_name,
            step_id,
            workflow_id,
        )
        return record

    async def get_approval_request(self, approval_id: str) -> ApprovalRequestRecord | None:
        """Retrieve an approval record by its unique ID."""
        return await self.store.get(approval_id)

    async def list_pending_approvals(
        self, workflow_id: str | None = None
    ) -> list[ApprovalRequestRecord]:
        """List all active, non-expired pending approval requests."""
        return await self.store.list_pending(workflow_id=workflow_id)

    async def find_request_for_step(
        self,
        workflow_id: str,
        step_id: str,
        thread_id: str | None = None,
        tool_name: str | None = None,
    ) -> ApprovalRequestRecord | None:
        """Find an existing approval request for a given workflow step."""
        all_records = await self.store.list_all()
        matches = [
            rec
            for rec in all_records
            if rec.workflow_id == workflow_id
            and rec.step_id == step_id
            and (thread_id is None or rec.thread_id == thread_id)
            and (tool_name is None or rec.tool_name == tool_name)
        ]
        if matches:
            matches.sort(key=lambda r: r.created_at, reverse=True)
            return matches[0]
        return None

    async def approve_request(
        self,
        approval_id: str,
        decided_by: str,
        reason: str | None = None,
    ) -> tuple[ApprovalRequestRecord, str]:
        """Atomically record an authenticated human operator's approval decision.

        Returns:
            (updated_record, decision_token)

        Raises:
            ApprovalError: If decision is invalid, request not found, expired, or already decided.
        """
        is_human, actor_err = validate_human_actor(decided_by)
        if not is_human:
            raise ApprovalError(actor_err or "Invalid human decision maker.")

        record = await self.store.get(approval_id)
        if not record:
            raise ApprovalError(f"Approval request '{approval_id}' was not found.")

        if record.status != ApprovalStatus.PENDING:
            raise ApprovalError(
                f"Approval request '{approval_id}' cannot be approved: "
                f"status is {record.status.value}."
            )

        if record.is_expired():
            await self.store.update_status(
                approval_id=approval_id,
                expected_status=ApprovalStatus.PENDING,
                new_status=ApprovalStatus.EXPIRED,
            )
            raise ApprovalError(f"Approval request '{approval_id}' has expired.")

        token = generate_approval_binding_token(
            approval_id=record.approval_id,
            workflow_id=record.workflow_id,
            thread_id=record.thread_id,
            step_id=record.step_id,
            tool_name=record.tool_name,
            arguments_hash=record.arguments_hash,
            permission_level=record.permission_level,
            request_version=record.request_version,
            secret_key=self.secret_key,
        )

        decision = ApprovalDecisionRecord(
            approval_id=approval_id,
            approved=True,
            decided_by=decided_by.strip(),
            reason=reason,
            decision_token=token,
        )

        success = await self.store.update_status(
            approval_id=approval_id,
            expected_status=ApprovalStatus.PENDING,
            new_status=ApprovalStatus.APPROVED,
            decision=decision,
            binding_token=token,
        )

        if not success:
            raise ApprovalError(
                f"Concurrent decision conflict: approval request '{approval_id}' "
                f"was modified or decided upon concurrently."
            )

        updated = await self.store.get(approval_id)
        if not updated:
            raise ApprovalError("Failed to retrieve record after approval update.")

        logger.info(
            "Approval '%s' granted by operator '%s' for tool '%s'",
            approval_id,
            decided_by,
            record.tool_name,
        )
        return updated, token

    async def reject_request(
        self,
        approval_id: str,
        decided_by: str,
        reason: str | None = None,
    ) -> ApprovalRequestRecord:
        """Atomically record an authenticated human operator's rejection decision."""
        is_human, actor_err = validate_human_actor(decided_by)
        if not is_human:
            raise ApprovalError(actor_err or "Invalid human decision maker.")

        record = await self.store.get(approval_id)
        if not record:
            raise ApprovalError(f"Approval request '{approval_id}' was not found.")

        if record.status != ApprovalStatus.PENDING:
            raise ApprovalError(
                f"Approval request '{approval_id}' cannot be rejected: "
                f"status is {record.status.value}."
            )

        decision = ApprovalDecisionRecord(
            approval_id=approval_id,
            approved=False,
            decided_by=decided_by.strip(),
            reason=reason,
        )

        success = await self.store.update_status(
            approval_id=approval_id,
            expected_status=ApprovalStatus.PENDING,
            new_status=ApprovalStatus.REJECTED,
            decision=decision,
        )

        if not success:
            raise ApprovalError(
                f"Concurrent decision conflict: approval request '{approval_id}' "
                f"was modified concurrently."
            )

        updated = await self.store.get(approval_id)
        if not updated:
            raise ApprovalError("Failed to retrieve record after rejection update.")

        logger.info("Approval '%s' rejected by operator '%s'", approval_id, decided_by)
        return updated

    async def expire_request(self, approval_id: str) -> ApprovalRequestRecord:
        """Explicitly transition an approval request to EXPIRED status."""
        record = await self.store.get(approval_id)
        if not record:
            raise ApprovalError(f"Approval request '{approval_id}' was not found.")

        await self.store.update_status(
            approval_id=approval_id,
            expected_status=record.status,
            new_status=ApprovalStatus.EXPIRED,
        )
        updated = await self.store.get(approval_id)
        return updated or record

    async def consume_approval(
        self,
        approval_id: str,
        workflow_id: str,
        thread_id: str,
        step_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        permission_level: PermissionLevel | None = None,
    ) -> tuple[bool, str | None]:
        """Atomically validate and single-use consume an approval record for tool execution.

        Enforces that:
        1. Approval exists and status is APPROVED
        2. Exact arguments hash, workflow, thread, step, tool, and permission match
        3. Expiration time has not passed
        4. Status transitions to CONSUMED atomically (prevents duplicate execution)

        Returns:
            (is_consumed, failure_reason)
        """
        record = await self.store.get(approval_id)
        if not record:
            return False, f"Approval request '{approval_id}' does not exist."

        # Verify cryptographic binding, tamper resistance, and validity
        is_valid, reason = verify_approval_binding(
            record=record,
            workflow_id=workflow_id,
            thread_id=thread_id,
            step_id=step_id,
            tool_name=tool_name,
            candidate_arguments=arguments,
            candidate_permission_level=permission_level,
            secret_key=self.secret_key,
        )
        if not is_valid:
            return False, reason

        # Atomically mark as consumed
        now = utc_now()
        success = await self.store.update_status(
            approval_id=approval_id,
            expected_status=ApprovalStatus.APPROVED,
            new_status=ApprovalStatus.CONSUMED,
            consumed_by=f"{workflow_id}:{step_id}",
            consumed_at=now,
        )

        if not success:
            return (
                False,
                f"Concurrent consumption error: approval '{approval_id}' was already "
                f"consumed by another execution (single-use protection).",
            )

        logger.info(
            "Approval '%s' successfully consumed for step '%s' (tool: %s)",
            approval_id,
            step_id,
            tool_name,
        )
        return True, None


_default_approval_service: ApprovalService | None = None


def get_approval_service() -> ApprovalService:
    """Return a singleton instance of the ApprovalService."""
    global _default_approval_service
    if _default_approval_service is None:
        _default_approval_service = ApprovalService()
    return _default_approval_service


def set_approval_service(service: ApprovalService | None) -> None:
    """Override the global approval service singleton (useful in test fixtures)."""
    global _default_approval_service
    _default_approval_service = service
