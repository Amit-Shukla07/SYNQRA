"""Durable Workflow Recovery Service for SYNQRA (Phase 9.5.2).

Inspects persisted LangGraph checkpoints, matches interrupted execution states
against the Approval Service, validates authorization consistency across restarts,
and safely resumes interrupted workflows without duplicating side-effects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from synqra.agents.approval.auth import OperatorIdentity, get_approval_authenticator
from synqra.agents.approval.service import ApprovalService, get_approval_service
from synqra.agents.checkpointer import get_checkpointer
from synqra.agents.exceptions import WorkflowRecoveryError
from synqra.agents.graph import create_agent_graph, resume_agent_workflow
from synqra.agents.schemas import ApprovalStatus, WorkflowStatus

logger = logging.getLogger("synqra.agents.workflow_recovery")


@dataclass
class RecoveredThreadStatus:
    """Diagnostic state summary for a persisted workflow thread."""

    thread_id: str
    workflow_id: str
    workflow_status: WorkflowStatus
    current_step_id: str | None
    is_interrupted: bool
    is_completed: bool
    has_pending_approval: bool
    approval_id: str | None
    tool_name: str | None
    checkpoint_id: str | None
    error_message: str | None = None


class WorkflowRecoveryService:
    """Recovers, audits, and resumes interrupted agent workflows across restarts."""

    def __init__(
        self,
        checkpointer: BaseCheckpointSaver | None = None,
        approval_service: ApprovalService | None = None,
        llm: Any | None = None,
    ) -> None:
        self._checkpointer = checkpointer
        self._approval_service = approval_service
        self._llm = llm

    async def _get_checkpointer(self) -> BaseCheckpointSaver:
        if self._checkpointer is not None:
            return self._checkpointer
        return await get_checkpointer()

    def _get_approval_service(self) -> ApprovalService:
        if self._approval_service is not None:
            return self._approval_service
        return get_approval_service()

    async def inspect_thread(
        self,
        thread_id: str,
        graph: CompiledStateGraph | None = None,
        operator: OperatorIdentity | None = None,
    ) -> RecoveredThreadStatus:
        """Inspect the latest persisted checkpoint for a thread and cross-reference approval state.

        Args:
            thread_id: Thread ID to inspect.
            graph: Optional compiled graph instance.
            operator: Mandatory authenticated operator identity for authorization enforcement.

        Returns:
            RecoveredThreadStatus with verified metadata.

        Raises:
            WorkflowRecoveryError: If operator is missing, lacks workflow permission,
                or checkpoint is invalid.
        """
        if operator is None:
            raise WorkflowRecoveryError(
                "Authentication required: operator identity is mandatory "
                "to inspect workflow threads."
            )

        cp = await self._get_checkpointer()
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        cp_tuple = await cp.aget_tuple(config)

        if cp_tuple is None or not cp_tuple.checkpoint:
            raise WorkflowRecoveryError(
                f"No persisted checkpoint found for thread '{thread_id}'. "
                "The workflow may not have started or checkpoint data is unavailable."
            )

        channel_values = dict(cp_tuple.checkpoint.get("channel_values", {}))
        checkpoint_id = cp_tuple.checkpoint.get("id")

        is_interrupted = False
        approval_id: str | None = None
        tool_name: str | None = None

        # Inspect pregel tasks from graph state snapshot if available
        try:
            from synqra.agents.llm import MockChatModel

            app = graph or create_agent_graph(llm=self._llm or MockChatModel(), checkpointer=cp)
            state_snap = await app.aget_state(config)
            if state_snap and state_snap.values:
                channel_values.update(state_snap.values)
            if state_snap and state_snap.tasks:
                for task in state_snap.tasks:
                    if task.interrupts:
                        is_interrupted = True
                        int_obj = task.interrupts[0]
                        int_val = int_obj.value if hasattr(int_obj, "value") else int_obj
                        if isinstance(int_val, dict):
                            approval_id = int_val.get("approval_id")
                            tool_name = int_val.get("tool_name")
                            if "workflow_id" in int_val:
                                channel_values.setdefault("workflow_id", int_val["workflow_id"])
                            if "step_id" in int_val:
                                channel_values.setdefault("current_step_id", int_val["step_id"])
        except Exception as exc:
            logger.debug("Failed to inspect pregel tasks from state graph: %s", exc)

        # Also check channel_values for __interrupt__ or approval_requests before auth check
        current_step_id = channel_values.get("current_step_id")
        workflow_id = channel_values.get("workflow_id", "wf_unknown")
        if not approval_id:
            interrupts = channel_values.get("__interrupt__", [])
            if interrupts:
                is_interrupted = True
                int_obj = interrupts[0]
                int_val = int_obj.value if hasattr(int_obj, "value") else int_obj
                if isinstance(int_val, dict):
                    approval_id = int_val.get("approval_id")
                    tool_name = int_val.get("tool_name")
                    if not current_step_id:
                        current_step_id = int_val.get("step_id")
                    if workflow_id == "wf_unknown" and "workflow_id" in int_val:
                        workflow_id = int_val["workflow_id"]

        if not approval_id:
            appr_reqs = channel_values.get("approval_requests", [])
            if appr_reqs:
                latest_req = appr_reqs[-1]
                approval_id = getattr(latest_req, "approval_id", None)
                tool_name = getattr(latest_req, "tool_name", None)

        # Mandatory workflow-level authorization check for the operator
        auth = get_approval_authenticator()
        ok, err = auth.authorize_operator(operator, workflow_id=workflow_id)
        if not ok:
            raise WorkflowRecoveryError(
                f"Access denied: operator '{operator.operator_id}' is not authorized "
                f"to inspect workflow '{workflow_id}': {err}"
            )

        raw_status = channel_values.get("workflow_status", WorkflowStatus.PENDING)
        status = WorkflowStatus(raw_status) if isinstance(raw_status, str) else raw_status

        is_completed = status in (
            WorkflowStatus.COMPLETED,
            WorkflowStatus.FAILED,
            WorkflowStatus.CANCELLED,
        )
        has_pending_approval = False

        if approval_id:
            appr_svc = self._get_approval_service()
            rec = await appr_svc.get_approval_request(approval_id)
            if rec:
                # Verify that approval record matches thread and workflow identifiers
                if rec.thread_id != thread_id:
                    raise WorkflowRecoveryError(
                        f"Approval mismatch: approval '{approval_id}' thread '{rec.thread_id}' "
                        f"does not match checkpoint thread '{thread_id}'."
                    )
                if rec.workflow_id != workflow_id:
                    raise WorkflowRecoveryError(
                        f"Approval mismatch: approval '{approval_id}' workflow '{rec.workflow_id}' "
                        f"does not match checkpoint workflow '{workflow_id}'."
                    )
                rec_ok, rec_err = auth.authorize_operator(operator, workflow_id=rec.workflow_id)
                if not rec_ok:
                    raise WorkflowRecoveryError(
                        f"Access denied: operator '{operator.operator_id}' is not authorized "
                        f"for approval workflow '{rec.workflow_id}': {rec_err}"
                    )
                has_pending_approval = rec.status == ApprovalStatus.PENDING
                if rec.status in (ApprovalStatus.APPROVED, ApprovalStatus.PENDING):
                    is_interrupted = True

        return RecoveredThreadStatus(
            thread_id=thread_id,
            workflow_id=workflow_id,
            workflow_status=status,
            current_step_id=current_step_id,
            is_interrupted=is_interrupted or (status == WorkflowStatus.AWAITING_APPROVAL),
            is_completed=is_completed,
            has_pending_approval=has_pending_approval,
            approval_id=approval_id,
            tool_name=tool_name,
            checkpoint_id=checkpoint_id,
        )

    async def recover_and_resume_workflow(
        self,
        thread_id: str,
        decision: dict[str, Any] | None = None,
        graph: CompiledStateGraph | None = None,
        llm: Any | None = None,
        operator: OperatorIdentity | None = None,
    ) -> dict[str, Any]:
        """Safely resume an interrupted workflow from its persisted checkpoint.

        Args:
            thread_id: Thread ID of the interrupted workflow.
            decision: Human operator's decision dict containing 'approved', 'decided_by', 'reason'.
            graph: Optional compiled graph instance.
            llm: Optional ChatModel instance for recovery graph construction.
            operator: Mandatory authenticated operator identity for authorization enforcement.

        Returns:
            Resumed workflow state dictionary.

        Raises:
            WorkflowRecoveryError: If operator is missing, unauthorized,
                or recovery preconditions fail.
        """
        if operator is None:
            raise WorkflowRecoveryError(
                "Authentication required: operator identity is mandatory "
                "to recover and resume workflows."
            )

        thread_status = await self.inspect_thread(thread_id, graph=graph, operator=operator)

        # Enforce workflow-level authorization and operator role for workflow resumption
        auth = get_approval_authenticator()
        ok, err = auth.authorize_operator(
            operator, workflow_id=thread_status.workflow_id, required_role="operator"
        )
        if not ok:
            raise WorkflowRecoveryError(
                f"Access denied: operator '{operator.operator_id}' is not authorized "
                f"to resume workflow '{thread_status.workflow_id}': {err}"
            )
        if decision and decision.get("decided_by"):
            if decision["decided_by"] != operator.operator_id:
                raise WorkflowRecoveryError(
                    f"Impersonation rejected: authenticated operator is "
                    f"'{operator.operator_id}', but decision asserted identity "
                    f"'{decision['decided_by']}'."
                )
        elif decision is not None:
            decision.setdefault("decided_by", operator.operator_id)

        # 1. Prevent resuming completed workflows
        if thread_status.is_completed:
            raise WorkflowRecoveryError(
                f"Workflow in thread '{thread_id}' is already "
                f"{thread_status.workflow_status.value}. "
                "Completed workflows cannot be resumed."
            )

        # 2. Check approval requirement
        appr_svc = self._get_approval_service()
        if thread_status.approval_id:
            rec = await appr_svc.get_approval_request(thread_status.approval_id)
            if not rec:
                raise WorkflowRecoveryError(
                    f"Approval record '{thread_status.approval_id}' referenced by checkpoint "
                    f"for thread '{thread_id}' was not found in the approval store."
                )

            # If rejected in store, reject resumption
            if rec.status == ApprovalStatus.REJECTED:
                # Can resume with rejection decision to conclude graph cleanly
                decision = decision or {
                    "approved": False,
                    "decided_by": rec.decision.decided_by if rec.decision else "system",
                    "reason": "Previously rejected in approval store.",
                }
            elif rec.status == ApprovalStatus.EXPIRED:
                raise WorkflowRecoveryError(
                    f"Cannot resume thread '{thread_id}': approval '{thread_status.approval_id}' "
                    f"expired at {rec.expires_at.isoformat()}."
                )
            elif rec.status == ApprovalStatus.CONSUMED:
                raise WorkflowRecoveryError(
                    f"Cannot resume thread '{thread_id}': approval '{thread_status.approval_id}' "
                    "was already consumed. Replay of consumed approval is strictly prohibited."
                )
            elif rec.status == ApprovalStatus.PENDING:
                if not decision:
                    raise WorkflowRecoveryError(
                        f"Cannot resume thread '{thread_id}': approval "
                        f"'{thread_status.approval_id}' is PENDING human decision. "
                        "A valid approval decision is required."
                    )
                # If caller provided an approved decision, ensure it is recorded in the store
                if decision.get("approved"):
                    # The approval record must be approved in the store first
                    raise WorkflowRecoveryError(
                        f"Cannot resume thread '{thread_id}': approval "
                        f"'{thread_status.approval_id}' is still PENDING in the approval store. "
                        "Approve the request via ApprovalService before resuming execution."
                    )

        if decision is None:
            raise WorkflowRecoveryError(
                f"Cannot resume thread '{thread_id}' without an explicit decision."
            )

        # 3. Resume through LangGraph using the same checkpointer
        cp = await self._get_checkpointer()
        active_llm = llm or self._llm
        app = graph or create_agent_graph(llm=active_llm, checkpointer=cp)

        logger.info(
            "Resuming workflow thread '%s' (workflow: %s, step: %s)",
            thread_id,
            thread_status.workflow_id,
            thread_status.current_step_id,
        )

        return await resume_agent_workflow(
            thread_id=thread_id,
            decision=decision,
            graph=app,
            checkpointer=cp,
        )

    async def inspect_thread_unauthenticated_test_only(
        self,
        thread_id: str,
        graph: CompiledStateGraph | None = None,
    ) -> RecoveredThreadStatus:
        """TEST-ONLY helper for inspecting thread checkpoints without operator credentials.

        WARNING: Strictly for automated unit/integration tests. Never invoke in production.
        """
        test_op = OperatorIdentity(operator_id="__test_internal__", roles=["admin"])
        return await self.inspect_thread(thread_id=thread_id, graph=graph, operator=test_op)

    async def recover_and_resume_workflow_unauthenticated_test_only(
        self,
        thread_id: str,
        decision: dict[str, Any] | None = None,
        graph: CompiledStateGraph | None = None,
        llm: Any | None = None,
    ) -> dict[str, Any]:
        """TEST-ONLY helper for resuming workflows without external operator credentials.

        WARNING: Strictly for automated unit/integration tests. Never invoke in production.
        """
        test_op = OperatorIdentity(operator_id="__test_internal__", roles=["admin"])
        return await self.recover_and_resume_workflow(
            thread_id=thread_id,
            decision=decision,
            graph=graph,
            llm=llm,
            operator=test_op,
        )
