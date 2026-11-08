"""Safety and Idempotency Tests for LangGraph Checkpoint & Approval Integration (Phase 9.5.2).

Verifies critical safety constraints:
1. Replay prevention for consumed approval records.
2. Prevention of execution when an approval was rejected.
3. Prevention of execution when an approval expired.
4. Prevention of execution when arguments were modified post-approval.
5. Prevention of automatic retry when a WRITE/SENSITIVE execution outcome is ambiguous.
6. Prevention of duplicate execution when a successful tool result is already persisted.
7. End-to-end recovery cycle against live PostgreSQL (SqlAlchemyApprovalStore + AsyncPostgresSaver).
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any

import pytest
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from synqra.agents.approval.auth import OperatorIdentity
from synqra.agents.approval.service import ApprovalService, get_approval_service
from synqra.agents.approval.sql_store import SqlAlchemyApprovalStore
from synqra.agents.approval.store import InMemoryApprovalStore
from synqra.agents.exceptions import WorkflowRecoveryError
from synqra.agents.executor import executor_node
from synqra.agents.graph import create_agent_graph
from synqra.agents.llm import MockChatModel
from synqra.agents.schemas import (
    ApprovalStatus,
    ExecutionPlan,
    PermissionLevel,
    PlanStep,
    ToolExecutionResult,
    WorkflowStatus,
)
from synqra.agents.tool_adapter import AgentToolAdapter
from synqra.agents.workflow_recovery import WorkflowRecoveryService

POSTGRES_TEST_URL = "postgresql://amitshukla@localhost:5432/synqra_test"
POSTGRES_SQLALCHEMY_URL = "postgresql+psycopg://amitshukla@localhost:5432/synqra_test"


async def check_postgres_alive() -> bool:
    try:
        pool = AsyncConnectionPool(POSTGRES_TEST_URL, open=False, min_size=1, max_size=2)
        await pool.open(wait=True)
        await pool.close()
        return True
    except Exception:
        return False


@pytest.mark.asyncio
async def test_consumed_approval_cannot_be_replayed() -> None:
    """Verify that a consumed approval record cannot execute an operation a second time."""
    store = InMemoryApprovalStore()
    appr_service = ApprovalService(store=store)

    rec = await appr_service.create_approval_request(
        workflow_id="wf_replay",
        thread_id="th_replay",
        step_id="step_1",
        tool_name="send_gmail_message",
        arguments={"recipient": "user@example.com", "subject": "Test"},
        permission_level=PermissionLevel.WRITE,
    )
    # Approve request
    await appr_service.approve_request(rec.approval_id, decided_by="admin")

    # First consumption succeeds
    consumed, _ = await appr_service.consume_approval(
        approval_id=rec.approval_id,
        workflow_id="wf_replay",
        thread_id="th_replay",
        step_id="step_1",
        tool_name="send_gmail_message",
        arguments={"recipient": "user@example.com", "subject": "Test"},
        permission_level=PermissionLevel.WRITE,
    )
    assert consumed is True

    # Replay attempt fails closed
    consumed_again, err = await appr_service.consume_approval(
        approval_id=rec.approval_id,
        workflow_id="wf_replay",
        thread_id="th_replay",
        step_id="step_1",
        tool_name="send_gmail_message",
        arguments={"recipient": "user@example.com", "subject": "Test"},
        permission_level=PermissionLevel.WRITE,
    )
    assert consumed_again is False
    assert "consumed" in err.lower()


@pytest.mark.asyncio
async def test_rejected_approval_cannot_execute() -> None:
    """Verify that if an approval is REJECTED in the store, the executor blocks execution."""
    appr_svc = get_approval_service()
    step_id = "step_rej_1"
    wid = f"wf_rej_{uuid.uuid4().hex[:6]}"
    tid = f"th_rej_{uuid.uuid4().hex[:6]}"

    rec = await appr_svc.create_approval_request(
        workflow_id=wid,
        thread_id=tid,
        step_id=step_id,
        tool_name="send_gmail_message",
        arguments={"recipient": "user@example.com"},
        permission_level=PermissionLevel.WRITE,
    )
    await appr_svc.reject_request(rec.approval_id, decided_by="admin", reason="Security policy")

    plan = ExecutionPlan(
        workflow_id=wid,
        goal="send email",
        user_request="send email",
        steps=[
            PlanStep(
                step_id=step_id,
                tool_name="send_gmail_message",
                arguments={"recipient": "user@example.com"},
                permission_level=PermissionLevel.WRITE,
                description="Test email",
            )
        ],
    )
    state = {
        "workflow_id": wid,
        "thread_id": tid,
        "execution_plan": plan,
        "completed_steps": [],
        "pending_steps": [step_id],
        "execution_results": {},
        "approval_requests": [],
        "error_history": [],
    }

    executed = False

    class MockAdapter(AgentToolAdapter):
        async def execute(
            self, tool: Any, arguments: dict[str, Any], timeout: float | None = None
        ) -> Any:
            nonlocal executed
            _ = tool
            _ = arguments
            _ = timeout
            executed = True
            return {"ok": True}

    adapter = MockAdapter()

    # Calling executor_node directly simulating resume with approved=True payload
    res = await executor_node(state=state, adapter=adapter)

    assert not executed
    assert res["workflow_status"] in (WorkflowStatus.AWAITING_APPROVAL, WorkflowStatus.REJECTED)


@pytest.mark.asyncio
async def test_expired_approval_cannot_execute() -> None:
    """Verify that an expired approval request is blocked from execution."""
    appr_svc = get_approval_service()
    step_id = "step_exp_1"
    wid = f"wf_exp_{uuid.uuid4().hex[:6]}"
    tid = f"th_exp_{uuid.uuid4().hex[:6]}"

    # Create request with very short expiration
    rec = await appr_svc.create_approval_request(
        workflow_id=wid,
        thread_id=tid,
        step_id=step_id,
        tool_name="send_gmail_message",
        arguments={"recipient": "user@example.com"},
        permission_level=PermissionLevel.WRITE,
    )
    await appr_svc.approve_request(rec.approval_id, decided_by="admin")

    # Force expiration by updating store directly
    req_entity = await appr_svc.store.get(rec.approval_id)
    if req_entity:
        req_entity.expires_at = datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=60)
        req_entity.status = ApprovalStatus.EXPIRED
        await appr_svc.store.save(req_entity)

    # Check that consuming returns expired
    consumed, err = await appr_svc.consume_approval(
        approval_id=rec.approval_id,
        workflow_id=wid,
        thread_id=tid,
        step_id=step_id,
        tool_name="send_gmail_message",
        arguments={"recipient": "user@example.com"},
        permission_level=PermissionLevel.WRITE,
    )
    assert not consumed
    assert "expired" in err.lower()


@pytest.mark.asyncio
async def test_modified_arguments_cannot_execute() -> None:
    """Verify that tampering with step arguments after approval causes execution to fail closed."""
    appr_svc = get_approval_service()
    step_id = "step_mod_1"
    wid = f"wf_mod_{uuid.uuid4().hex[:6]}"
    tid = f"th_mod_{uuid.uuid4().hex[:6]}"

    rec = await appr_svc.create_approval_request(
        workflow_id=wid,
        thread_id=tid,
        step_id=step_id,
        tool_name="send_gmail_message",
        arguments={"recipient": "alice@example.com", "amount": "10"},
        permission_level=PermissionLevel.WRITE,
    )
    await appr_svc.approve_request(rec.approval_id, decided_by="admin")

    # Attempt to consume with modified arguments
    consumed, err = await appr_svc.consume_approval(
        approval_id=rec.approval_id,
        workflow_id=wid,
        thread_id=tid,
        step_id=step_id,
        tool_name="send_gmail_message",
        arguments={"recipient": "attacker@example.com", "amount": "1000000"},  # Tampered!
        permission_level=PermissionLevel.WRITE,
    )
    assert not consumed
    assert "tampered" in err.lower() or "mismatch" in err.lower() or "hash" in err.lower()


@pytest.mark.asyncio
async def test_ambiguous_write_execution_not_retried() -> None:
    """Verify that if an approval is CONSUMED but no tool result persisted, execution fails."""
    appr_svc = get_approval_service()
    step_id = "step_ambig_1"
    wid = f"wf_ambig_{uuid.uuid4().hex[:6]}"
    tid = f"th_ambig_{uuid.uuid4().hex[:6]}"

    rec = await appr_svc.create_approval_request(
        workflow_id=wid,
        thread_id=tid,
        step_id=step_id,
        tool_name="send_gmail_message",
        arguments={"recipient": "client@example.com"},
        permission_level=PermissionLevel.WRITE,
    )
    await appr_svc.approve_request(rec.approval_id, decided_by="admin")
    # Consume it
    await appr_svc.consume_approval(
        approval_id=rec.approval_id,
        workflow_id=wid,
        thread_id=tid,
        step_id=step_id,
        tool_name="send_gmail_message",
        arguments={"recipient": "client@example.com"},
        permission_level=PermissionLevel.WRITE,
    )

    # Now verify in recovery service
    from langgraph.checkpoint.memory import MemorySaver

    saver = MemorySaver()
    config = {"configurable": {"thread_id": tid, "checkpoint_ns": ""}}
    checkpoint = {
        "v": 1,
        "ts": "2026-11-08T12:00:00.000000+00:00",
        "id": str(uuid.uuid4()),
        "channel_values": {
            "workflow_id": wid,
            "thread_id": tid,
            "workflow_status": WorkflowStatus.AWAITING_APPROVAL.value,
            "current_step_id": step_id,
            "__interrupt__": [
                {
                    "approval_id": rec.approval_id,
                    "workflow_id": wid,
                    "thread_id": tid,
                    "step_id": step_id,
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
    await saver.aput(
        config,
        checkpoint,
        {},
        {"workflow_id": 1, "thread_id": 1, "workflow_status": 1, "__interrupt__": 1},
    )

    recovery_svc = WorkflowRecoveryService(checkpointer=saver, approval_service=appr_svc)
    admin_op = OperatorIdentity(operator_id="admin_tester", roles=["admin"])

    # In recovery service, attempting to resume an already consumed approval fails
    with pytest.raises(WorkflowRecoveryError, match="already consumed"):
        await recovery_svc.recover_and_resume_workflow(
            thread_id=tid,
            decision={"approved": True},
            operator=admin_op,
        )


@pytest.mark.asyncio
async def test_persisted_successful_result_not_reexecuted() -> None:
    """Verify that a step with a persisted successful result is reused without re-execution."""
    step_id = "step_done_1"
    wid = "wf_already_done"
    tid = "th_already_done"

    plan = ExecutionPlan(
        workflow_id=wid,
        goal="run test",
        user_request="run test",
        steps=[
            PlanStep(
                step_id=step_id,
                tool_name="health_check",
                arguments={},
                permission_level=PermissionLevel.READ,
                description="Check system health",
            )
        ],
    )
    already_persisted_result = ToolExecutionResult(
        tool_name="health_check",
        arguments={},
        success=True,
        result={"status": "previously_completed_and_cached"},
    )
    state = {
        "workflow_id": wid,
        "thread_id": tid,
        "execution_plan": plan,
        "completed_steps": [step_id],
        "pending_steps": [],
        "execution_results": {step_id: already_persisted_result},
        "approval_requests": [],
        "error_history": [],
    }

    tool_call_count = 0

    class CountingAdapter(AgentToolAdapter):
        async def execute(
            self, tool: Any, arguments: dict[str, Any], timeout: float | None = None
        ) -> Any:
            nonlocal tool_call_count
            _ = tool
            _ = arguments
            _ = timeout
            tool_call_count += 1
            return {"status": "freshly_executed"}

    adapter = CountingAdapter()
    res = await executor_node(state=state, adapter=adapter)

    # Tool was NOT called again
    assert tool_call_count == 0
    assert res["workflow_status"] == WorkflowStatus.COMPLETED


@pytest.mark.asyncio
async def test_live_postgres_interrupted_recovery_cycle() -> None:
    """Live PostgreSQL End-to-End Test:

    1. Initialize SqlAlchemyApprovalStore on live PostgreSQL.
    2. Initialize AsyncPostgresSaver on live PostgreSQL.
    3. Run agent workflow requiring human approval; verify interruption and persistence.
    4. Dispose graph and simulate server restart with brand new graph and session.
    5. Approve request in PostgreSQL approval store.
    6. Resume workflow from PostgreSQL checkpoint; verify execution completes cleanly.
    """
    alive = await check_postgres_alive()
    if not alive:
        pytest.skip("PostgreSQL live test database not available.")

    # 1. Setup Live PostgreSQL approval store
    from synqra.agents.approval.service import set_approval_service

    engine = create_async_engine(POSTGRES_SQLALCHEMY_URL, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    sql_store = SqlAlchemyApprovalStore(session_maker=session_factory)
    appr_service = ApprovalService(store=sql_store)
    set_approval_service(appr_service)

    # 2. Setup Live PostgreSQL checkpointer
    pool = AsyncConnectionPool(
        POSTGRES_TEST_URL,
        open=False,
        min_size=1,
        max_size=3,
        kwargs={"autocommit": True, "row_factory": dict_row},
    )
    await pool.open(wait=True)

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    saver = AsyncPostgresSaver(conn=pool)
    await saver.setup()

    tid = f"th_live_cycle_{uuid.uuid4().hex[:8]}"
    wid = f"wf_live_cycle_{uuid.uuid4().hex[:8]}"

    # Tool execution mock to record calls
    executed_tools: list[str] = []

    class MockLiveToolAdapter(AgentToolAdapter):
        async def execute(
            self,
            tool: Any,
            arguments: dict[str, Any],
            timeout: float | None = None,
        ) -> Any:
            _ = arguments
            _ = timeout
            executed_tools.append(tool.name)
            return {"status": "dispatched", "tool": tool.name}

    adapter = MockLiveToolAdapter()

    from synqra.agents.planner import PlannerOutput, PlanStepProposal

    llm = MockChatModel(
        structured_response=PlannerOutput(
            reasoning_summary="Send alert email to admin",
            steps=[
                PlanStepProposal(
                    step_id="step_write_1",
                    tool_name="send_gmail_message",
                    arguments={
                        "recipient": "admin@example.com",
                        "subject": "Alert",
                        "confirm": True,
                    },
                    description="Send alert email",
                )
            ],
        ),
    )

    try:
        # Graph Instance 1 (Pre-restart)
        graph1 = create_agent_graph(llm=llm, adapter=adapter, checkpointer=saver)

        initial_input = {
            "workflow_id": wid,
            "user_request": "send alert email to admin@example.com",
            "thread_id": tid,
            "workflow_status": WorkflowStatus.PENDING,
        }
        config = {"configurable": {"thread_id": tid, "checkpoint_ns": ""}}
        res1 = await graph1.ainvoke(initial_input, config=config)

        # Interrupted on human approval
        assert "__interrupt__" in res1
        int_val = (
            res1["__interrupt__"][0].value
            if hasattr(res1["__interrupt__"][0], "value")
            else res1["__interrupt__"][0]
        )
        appr_id = int_val["approval_id"]

        # Verify checkpoint in PostgreSQL
        cp_tuple = await saver.aget_tuple(config)
        assert cp_tuple is not None
        assert cp_tuple.checkpoint["channel_values"]["workflow_id"] == wid

        # 4. SIMULATE SERVER RESTART: Create brand new Graph Instance 2 and new session
        # Human approves in PostgreSQL store
        await appr_service.approve_request(
            appr_id, decided_by="super_admin", reason="Live PostgreSQL approval"
        )

        graph2 = create_agent_graph(llm=llm, adapter=adapter, checkpointer=saver)
        recovery_svc = WorkflowRecoveryService(checkpointer=saver, approval_service=appr_service)
        admin_op = OperatorIdentity(operator_id="super_admin", roles=["admin"])

        # Verify recovery service thread inspection
        thread_stat = await recovery_svc.inspect_thread(tid, operator=admin_op)
        assert thread_stat.thread_id == tid
        assert thread_stat.workflow_id == wid
        assert thread_stat.approval_id == appr_id

        # Resume workflow from PostgreSQL checkpoint
        res2 = await recovery_svc.recover_and_resume_workflow(
            thread_id=tid,
            decision={
                "approved": True,
                "decided_by": "super_admin",
                "reason": "Live PostgreSQL approval",
            },
            graph=graph2,
            operator=admin_op,
        )

        assert res2["workflow_status"] in (WorkflowStatus.COMPLETED, WorkflowStatus.EXECUTING)
        assert "send_gmail_message" in executed_tools
        assert "step_write_1" in res2["completed_steps"]

        # Verify that attempting to resume again is rejected as completed
        with pytest.raises(WorkflowRecoveryError, match="already completed"):
            await recovery_svc.recover_and_resume_workflow(
                thread_id=tid,
                decision={"approved": True, "decided_by": "super_admin"},
                graph=graph2,
                operator=admin_op,
            )
    finally:
        set_approval_service(None)
        await pool.close()
        await engine.dispose()
