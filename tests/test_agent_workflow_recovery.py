"""Tests for Durable Workflow Recovery Service (Phase 9.5.2).

Validates inspection of persisted checkpoints, cross-referencing with ApprovalService,
clean resumption across graph instances, and fail-closed handling of completed,
missing, or unauthorized workflows.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver

from synqra.agents.approval.auth import OperatorIdentity
from synqra.agents.approval.service import ApprovalService
from synqra.agents.approval.store import InMemoryApprovalStore
from synqra.agents.exceptions import WorkflowRecoveryError
from synqra.agents.graph import create_agent_graph
from synqra.agents.llm import MockChatModel
from synqra.agents.schemas import (
    PermissionLevel,
    WorkflowStatus,
)
from synqra.agents.workflow_recovery import (
    RecoveredThreadStatus,
    WorkflowRecoveryService,
)


@pytest.fixture
def admin_operator() -> OperatorIdentity:
    return OperatorIdentity(operator_id="admin_op", roles=["admin"])


@pytest.fixture
def memory_approval_service() -> ApprovalService:
    store = InMemoryApprovalStore()
    return ApprovalService(store=store)


@pytest.fixture
def memory_checkpointer() -> MemorySaver:
    return MemorySaver()


@pytest.mark.asyncio
async def test_recovery_inspect_thread_pending_approval(
    memory_checkpointer: MemorySaver,
    memory_approval_service: ApprovalService,
    admin_operator: OperatorIdentity,
) -> None:
    """Test inspecting a thread that was interrupted waiting for human approval."""
    tid = f"th_rec_inspect_{uuid.uuid4().hex[:8]}"
    wid = f"wf_rec_inspect_{uuid.uuid4().hex[:8]}"

    # Create approval request in service
    rec = await memory_approval_service.create_approval_request(
        workflow_id=wid,
        thread_id=tid,
        step_id="step_1",
        tool_name="send_gmail_message",
        arguments={"to": "user@example.com", "subject": "Hello"},
        permission_level=PermissionLevel.WRITE,
        description="Send email notification",
    )

    # Save checkpoint simulating LangGraph interruption
    config = {"configurable": {"thread_id": tid, "checkpoint_ns": ""}}
    checkpoint = {
        "v": 1,
        "ts": "2026-11-08T12:00:00.000000+00:00",
        "id": str(uuid.uuid4()),
        "channel_values": {
            "workflow_id": wid,
            "thread_id": tid,
            "workflow_status": WorkflowStatus.AWAITING_APPROVAL.value,
            "current_step_id": "step_1",
            "__interrupt__": [
                {
                    "approval_id": rec.approval_id,
                    "workflow_id": wid,
                    "thread_id": tid,
                    "step_id": "step_1",
                    "tool_name": "send_gmail_message",
                }
            ],
        },
        "channel_versions": {
            "workflow_id": 1,
            "thread_id": 1,
            "workflow_status": 1,
            "__interrupt__": 1,
        },
        "versions_seen": {},
    }
    await memory_checkpointer.aput(
        config,
        checkpoint,
        {},
        {"workflow_id": 1, "thread_id": 1, "workflow_status": 1, "__interrupt__": 1},
    )

    recovery_svc = WorkflowRecoveryService(
        checkpointer=memory_checkpointer,
        approval_service=memory_approval_service,
    )

    status = await recovery_svc.inspect_thread(tid, operator=admin_operator)
    assert isinstance(status, RecoveredThreadStatus)
    assert status.thread_id == tid
    assert status.workflow_id == wid
    assert status.is_interrupted is True
    assert status.has_pending_approval is True
    assert status.approval_id == rec.approval_id
    assert status.tool_name == "send_gmail_message"
    assert status.is_completed is False


@pytest.mark.asyncio
async def test_recovery_refuses_completed_workflow(
    memory_checkpointer: MemorySaver,
    memory_approval_service: ApprovalService,
    admin_operator: OperatorIdentity,
) -> None:
    """Verify that a completed workflow cannot be recovered or re-executed."""
    tid = f"th_completed_{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": tid, "checkpoint_ns": ""}}

    checkpoint = {
        "v": 1,
        "ts": "2026-11-08T12:00:00.000000+00:00",
        "id": str(uuid.uuid4()),
        "channel_values": {
            "workflow_id": "wf_done",
            "thread_id": tid,
            "workflow_status": WorkflowStatus.COMPLETED.value,
        },
        "channel_versions": {
            "workflow_id": 1,
            "thread_id": 1,
            "workflow_status": 1,
            "__interrupt__": 1,
        },
        "versions_seen": {},
    }
    await memory_checkpointer.aput(
        config,
        checkpoint,
        {},
        {"workflow_id": 1, "thread_id": 1, "workflow_status": 1, "__interrupt__": 1},
    )

    recovery_svc = WorkflowRecoveryService(
        checkpointer=memory_checkpointer,
        approval_service=memory_approval_service,
    )

    with pytest.raises(WorkflowRecoveryError, match="already completed"):
        await recovery_svc.recover_and_resume_workflow(
            thread_id=tid,
            decision={"approved": True, "decided_by": "admin_op"},
            operator=admin_operator,
        )


@pytest.mark.asyncio
async def test_recovery_refuses_missing_checkpoint(
    memory_checkpointer: MemorySaver,
    memory_approval_service: ApprovalService,
    admin_operator: OperatorIdentity,
) -> None:
    """Verify that attempting to inspect or recover a non-existent thread fails closed."""
    recovery_svc = WorkflowRecoveryService(
        checkpointer=memory_checkpointer,
        approval_service=memory_approval_service,
    )

    with pytest.raises(WorkflowRecoveryError, match="No persisted checkpoint"):
        await recovery_svc.inspect_thread("th_never_existed", operator=admin_operator)

    with pytest.raises(WorkflowRecoveryError, match="No persisted checkpoint"):
        await recovery_svc.recover_and_resume_workflow(
            thread_id="th_never_existed",
            decision={"approved": True},
            operator=admin_operator,
        )


@pytest.mark.asyncio
async def test_recovery_refuses_when_approval_missing_from_store(
    memory_checkpointer: MemorySaver,
    memory_approval_service: ApprovalService,
    admin_operator: OperatorIdentity,
) -> None:
    """Verify that if checkpoint references a missing approval_id, it fails closed."""
    tid = f"th_missing_appr_{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": tid, "checkpoint_ns": ""}}

    checkpoint = {
        "v": 1,
        "ts": "2026-11-08T12:00:00.000000+00:00",
        "id": str(uuid.uuid4()),
        "channel_values": {
            "workflow_id": "wf_missing_appr",
            "thread_id": tid,
            "workflow_status": WorkflowStatus.AWAITING_APPROVAL.value,
            "__interrupt__": [
                {
                    "approval_id": "appr_does_not_exist",
                    "step_id": "step_1",
                    "tool_name": "send_gmail_message",
                }
            ],
        },
        "channel_versions": {
            "workflow_id": 1,
            "thread_id": 1,
            "workflow_status": 1,
            "__interrupt__": 1,
        },
        "versions_seen": {},
    }
    await memory_checkpointer.aput(
        config,
        checkpoint,
        {},
        {"workflow_id": 1, "thread_id": 1, "workflow_status": 1, "__interrupt__": 1},
    )

    recovery_svc = WorkflowRecoveryService(
        checkpointer=memory_checkpointer,
        approval_service=memory_approval_service,
    )

    with pytest.raises(WorkflowRecoveryError, match="was not found in the approval store"):
        await recovery_svc.recover_and_resume_workflow(
            thread_id=tid,
            decision={"approved": True, "decided_by": "admin_op"},
            operator=admin_operator,
        )


@pytest.mark.asyncio
async def test_recovery_refuses_when_approval_still_pending_in_store(
    memory_checkpointer: MemorySaver,
    memory_approval_service: ApprovalService,
    admin_operator: OperatorIdentity,
) -> None:
    """Verify that if caller passes approved=True but store is PENDING, recovery fails."""
    tid = f"th_still_pending_{uuid.uuid4().hex[:8]}"
    wid = f"wf_still_pending_{uuid.uuid4().hex[:8]}"

    rec = await memory_approval_service.create_approval_request(
        workflow_id=wid,
        thread_id=tid,
        step_id="step_1",
        tool_name="send_gmail_message",
        arguments={"to": "test@example.com"},
        permission_level=PermissionLevel.WRITE,
        description="Send email",
    )

    config = {"configurable": {"thread_id": tid, "checkpoint_ns": ""}}
    checkpoint = {
        "v": 1,
        "ts": "2026-11-08T12:00:00.000000+00:00",
        "id": str(uuid.uuid4()),
        "channel_values": {
            "workflow_id": wid,
            "thread_id": tid,
            "workflow_status": WorkflowStatus.AWAITING_APPROVAL.value,
            "__interrupt__": [
                {
                    "approval_id": rec.approval_id,
                    "workflow_id": wid,
                    "thread_id": tid,
                    "step_id": "step_1",
                    "tool_name": "send_gmail_message",
                }
            ],
        },
        "channel_versions": {
            "workflow_id": 1,
            "thread_id": 1,
            "workflow_status": 1,
            "__interrupt__": 1,
        },
        "versions_seen": {},
    }
    await memory_checkpointer.aput(
        config,
        checkpoint,
        {},
        {"workflow_id": 1, "thread_id": 1, "workflow_status": 1, "__interrupt__": 1},
    )

    recovery_svc = WorkflowRecoveryService(
        checkpointer=memory_checkpointer,
        approval_service=memory_approval_service,
    )

    with pytest.raises(WorkflowRecoveryError, match="still PENDING in the approval store"):
        await recovery_svc.recover_and_resume_workflow(
            thread_id=tid,
            decision={"approved": True, "decided_by": "admin_op"},
            operator=admin_operator,
        )


@pytest.mark.asyncio
async def test_recovery_resume_interrupted_workflow_across_instances(
    memory_checkpointer: MemorySaver,
) -> None:
    """Full lifecycle recovery test: pause on approval, approve in store, resume in graph 2."""
    from synqra.agents.planner import PlannerOutput, PlanStepProposal
    from synqra.agents.tool_adapter import AgentToolAdapter
    from synqra.agents.tools import get_default_agent_tool_registry

    tid = f"th_e2e_rec_{uuid.uuid4().hex[:8]}"
    wid = f"wf_e2e_rec_{uuid.uuid4().hex[:8]}"

    # Setup mock tool executor to avoid real writes
    executed_tools: list[str] = []

    class MockToolAdapter(AgentToolAdapter):
        async def execute(
            self,
            tool: Any,
            arguments: dict[str, Any],
            timeout: float | None = None,
        ) -> Any:
            _ = arguments
            _ = timeout
            executed_tools.append(tool.name)
            return {"status": "sent", "mock": True}

    adapter = MockToolAdapter()
    registry = get_default_agent_tool_registry()

    llm = MockChatModel(
        structured_response=PlannerOutput(
            reasoning_summary="Plan to send email notification",
            steps=[
                PlanStepProposal(
                    step_id="step_1",
                    tool_name="send_gmail_message",
                    arguments={
                        "recipient": "dev@example.com",
                        "subject": "Test",
                        "body": "Hi",
                        "confirm": True,
                    },
                    description="Send test notification email",
                )
            ],
        ),
    )

    # Initial graph instance
    graph1 = create_agent_graph(
        llm=llm,
        registry=registry,
        adapter=adapter,
        checkpointer=memory_checkpointer,
    )

    initial_input = {
        "workflow_id": wid,
        "user_request": "send an email to dev@example.com",
        "thread_id": tid,
        "workflow_status": WorkflowStatus.PENDING,
    }
    config = {"configurable": {"thread_id": tid, "checkpoint_ns": ""}}
    res1 = await graph1.ainvoke(initial_input, config=config)

    # Workflow is paused awaiting approval
    assert "__interrupt__" in res1
    int_val = (
        res1["__interrupt__"][0].value
        if hasattr(res1["__interrupt__"][0], "value")
        else res1["__interrupt__"][0]
    )
    appr_id = int_val["approval_id"]

    # In approval service, human approves
    from synqra.agents.approval.service import get_approval_service

    appr_svc = get_approval_service()
    await appr_svc.approve_request(
        appr_id, decided_by="sec_admin", reason="Authorized deployment email"
    )

    # Simulate server restart by creating a brand new graph instance with the same checkpointer
    graph2 = create_agent_graph(
        llm=llm,
        registry=registry,
        adapter=adapter,
        checkpointer=memory_checkpointer,
    )

    recovery_svc = WorkflowRecoveryService(
        checkpointer=memory_checkpointer,
        approval_service=appr_svc,
    )
    sec_op = OperatorIdentity(operator_id="sec_admin", roles=["admin"])

    # Inspect thread before resume
    status = await recovery_svc.inspect_thread(tid, operator=sec_op)
    assert status.is_interrupted is True
    assert status.approval_id == appr_id

    # Resume through recovery service
    res2 = await recovery_svc.recover_and_resume_workflow(
        thread_id=tid,
        decision={
            "approved": True,
            "decided_by": "sec_admin",
            "reason": "Authorized deployment email",
        },
        graph=graph2,
        operator=sec_op,
    )

    assert res2["workflow_status"] in (WorkflowStatus.COMPLETED, WorkflowStatus.EXECUTING)
    assert "send_gmail_message" in executed_tools
    assert "step_1" in res2["completed_steps"]


@pytest.mark.asyncio
async def test_recovery_authorization_restricted_operator(
    memory_checkpointer: MemorySaver,
    memory_approval_service: ApprovalService,
) -> None:
    """Verify that an operator restricted to wf_alpha cannot inspect or resume wf_beta."""
    from synqra.agents.approval.auth import OperatorIdentity

    tid = f"th_auth_test_{uuid.uuid4().hex[:8]}"
    wid = "wf_beta"

    rec = await memory_approval_service.create_approval_request(
        workflow_id=wid,
        thread_id=tid,
        step_id="step_1",
        tool_name="send_gmail_message",
        arguments={"to": "dev@example.com"},
        permission_level=PermissionLevel.WRITE,
    )

    config = {"configurable": {"thread_id": tid, "checkpoint_ns": ""}}
    checkpoint = {
        "v": 1,
        "ts": "2026-11-08T12:00:00.000000+00:00",
        "id": str(uuid.uuid4()),
        "channel_values": {
            "workflow_id": wid,
            "thread_id": tid,
            "workflow_status": WorkflowStatus.AWAITING_APPROVAL.value,
            "current_step_id": "step_1",
            "__interrupt__": [
                {
                    "approval_id": rec.approval_id,
                    "workflow_id": wid,
                    "thread_id": tid,
                    "step_id": "step_1",
                    "tool_name": "send_gmail_message",
                }
            ],
        },
        "channel_versions": {
            "workflow_id": 1,
            "thread_id": 1,
            "workflow_status": 1,
            "__interrupt__": 1,
        },
        "versions_seen": {},
    }
    await memory_checkpointer.aput(
        config,
        checkpoint,
        {},
        {"workflow_id": 1, "thread_id": 1, "workflow_status": 1, "__interrupt__": 1},
    )

    recovery_svc = WorkflowRecoveryService(
        checkpointer=memory_checkpointer,
        approval_service=memory_approval_service,
    )

    # Operator restricted strictly to wf_alpha
    restricted_operator = OperatorIdentity(
        operator_id="operator_bob",
        roles=["operator"],
        allowed_workflows=["wf_alpha"],
    )

    # Inspection is denied
    with pytest.raises(WorkflowRecoveryError, match="Access denied.*not authorized"):
        await recovery_svc.inspect_thread(tid, operator=restricted_operator)

    # Resumption is denied
    with pytest.raises(WorkflowRecoveryError, match="Access denied.*not authorized"):
        await recovery_svc.recover_and_resume_workflow(
            thread_id=tid,
            decision={"approved": True},
            operator=restricted_operator,
        )

    # Impersonation is rejected
    authorized_operator = OperatorIdentity(
        operator_id="operator_alice",
        roles=["operator"],
        allowed_workflows=[wid],
    )
    with pytest.raises(WorkflowRecoveryError, match="Impersonation rejected"):
        await recovery_svc.recover_and_resume_workflow(
            thread_id=tid,
            decision={"approved": True, "decided_by": "impostor_carl"},
            operator=authorized_operator,
        )


@pytest.mark.asyncio
async def test_recovery_missing_operator_rejected(
    memory_checkpointer: MemorySaver,
    memory_approval_service: ApprovalService,
) -> None:
    """Verify that omitting operator identity fails closed with an authorization error."""
    tid = f"th_no_op_{uuid.uuid4().hex[:8]}"
    wid = "wf_no_op"

    config = {"configurable": {"thread_id": tid, "checkpoint_ns": ""}}
    checkpoint = {
        "v": 1,
        "ts": "2026-11-08T12:00:00.000000+00:00",
        "id": str(uuid.uuid4()),
        "channel_values": {
            "workflow_id": wid,
            "thread_id": tid,
            "workflow_status": WorkflowStatus.AWAITING_APPROVAL.value,
        },
        "channel_versions": {"workflow_id": 1, "thread_id": 1, "workflow_status": 1},
        "versions_seen": {},
    }
    await memory_checkpointer.aput(
        config,
        checkpoint,
        {},
        {"workflow_id": 1, "thread_id": 1, "workflow_status": 1},
    )

    recovery_svc = WorkflowRecoveryService(
        checkpointer=memory_checkpointer,
        approval_service=memory_approval_service,
    )

    # 1. Missing operator on inspect_thread is rejected
    with pytest.raises(WorkflowRecoveryError, match="operator identity is mandatory"):
        await recovery_svc.inspect_thread(tid, operator=None)

    # 2. Missing operator on recover_and_resume_workflow is rejected
    with pytest.raises(WorkflowRecoveryError, match="operator identity is mandatory"):
        await recovery_svc.recover_and_resume_workflow(
            thread_id=tid,
            decision={"approved": True},
            operator=None,
        )


@pytest.mark.asyncio
async def test_recovery_unauthenticated_test_helpers(
    memory_checkpointer: MemorySaver,
    memory_approval_service: ApprovalService,
) -> None:
    """Verify that explicit test-only unauthenticated helpers work for test suites."""
    tid = f"th_test_helper_{uuid.uuid4().hex[:8]}"
    wid = "wf_test_helper"

    config = {"configurable": {"thread_id": tid, "checkpoint_ns": ""}}
    checkpoint = {
        "v": 1,
        "ts": "2026-11-08T12:00:00.000000+00:00",
        "id": str(uuid.uuid4()),
        "channel_values": {
            "workflow_id": wid,
            "thread_id": tid,
            "workflow_status": WorkflowStatus.COMPLETED.value,
        },
        "channel_versions": {"workflow_id": 1, "thread_id": 1, "workflow_status": 1},
        "versions_seen": {},
    }
    await memory_checkpointer.aput(
        config,
        checkpoint,
        {},
        {"workflow_id": 1, "thread_id": 1, "workflow_status": 1},
    )

    recovery_svc = WorkflowRecoveryService(
        checkpointer=memory_checkpointer,
        approval_service=memory_approval_service,
    )

    status = await recovery_svc.inspect_thread_unauthenticated_test_only(tid)
    assert status.workflow_id == wid
    assert status.is_completed is True


@pytest.mark.asyncio
async def test_recovery_approval_thread_and_workflow_mismatch_rejected(
    memory_checkpointer: MemorySaver,
    memory_approval_service: ApprovalService,
    admin_operator: OperatorIdentity,
) -> None:
    """Verify that approval records pointing to a different thread or workflow are rejected."""
    tid = f"th_mismatch_{uuid.uuid4().hex[:8]}"
    wid = "wf_legit"

    # Approval belongs to another thread
    rec_wrong_thread = await memory_approval_service.create_approval_request(
        workflow_id=wid,
        thread_id="th_different_one",
        step_id="step_1",
        tool_name="send_gmail_message",
        arguments={"to": "dev@example.com"},
        permission_level=PermissionLevel.WRITE,
    )

    config = {"configurable": {"thread_id": tid, "checkpoint_ns": ""}}
    checkpoint = {
        "v": 1,
        "ts": "2026-11-08T12:00:00.000000+00:00",
        "id": str(uuid.uuid4()),
        "channel_values": {
            "workflow_id": wid,
            "thread_id": tid,
            "workflow_status": WorkflowStatus.AWAITING_APPROVAL.value,
            "__interrupt__": [{"approval_id": rec_wrong_thread.approval_id}],
        },
        "channel_versions": {
            "workflow_id": 1,
            "thread_id": 1,
            "workflow_status": 1,
            "__interrupt__": 1,
        },
        "versions_seen": {},
    }
    await memory_checkpointer.aput(
        config,
        checkpoint,
        {},
        {"workflow_id": 1, "thread_id": 1, "workflow_status": 1, "__interrupt__": 1},
    )

    recovery_svc = WorkflowRecoveryService(
        checkpointer=memory_checkpointer,
        approval_service=memory_approval_service,
    )

    with pytest.raises(WorkflowRecoveryError, match="Approval mismatch.*thread"):
        await recovery_svc.inspect_thread(tid, operator=admin_operator)
