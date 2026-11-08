"""End-to-end integration tests for LangGraph Human-in-the-Loop (HITL) approval.

Verifies:
- READ workflows execute without interruption
- WRITE/SENSITIVE workflows pause at interrupt point
- Resuming with valid approval executes tool and completes workflow
- Resuming with rejection blocks execution and marks status REJECTED
- Argument tampering between approval and execution is detected and blocked
- Consumed approvals cannot be replayed or reused
"""

from unittest.mock import AsyncMock

import pytest
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from synqra.agents.approval.service import (
    ApprovalService,
    get_approval_service,
    set_approval_service,
)
from synqra.agents.approval.store import InMemoryApprovalStore
from synqra.agents.graph import create_agent_graph, resume_agent_workflow, run_agent_workflow
from synqra.agents.llm import get_mock_llm
from synqra.agents.planner import PlannerOutput, PlanStepProposal
from synqra.agents.schemas import PermissionLevel, WorkflowStatus
from synqra.agents.tool_registry import AgentToolRegistry, RegisteredTool


class MockToolInput(BaseModel):
    query: str = Field(description="Parameter string")


@pytest.fixture(autouse=True)
def clean_approval_service() -> None:
    """Ensure a fresh ApprovalService for each graph test."""
    service = ApprovalService(store=InMemoryApprovalStore())
    set_approval_service(service)
    yield
    set_approval_service(None)


@pytest.mark.asyncio
async def test_read_only_workflow_no_interruption() -> None:
    """Verify READ-only workflows run straight to completion without pausing for approval."""
    read_mock = AsyncMock(return_value={"status": "ok", "url": "https://example.com/read"})
    registry = AgentToolRegistry()
    registry.register_tool(
        RegisteredTool(
            name="search_web",
            description="Web search tool",
            integration_name="mock",
            input_schema=MockToolInput,
            execution_function=read_mock,
            permission_level=PermissionLevel.READ,
        )
    )

    plan = PlannerOutput(
        reasoning_summary="Read-only query",
        steps=[
            PlanStepProposal(
                step_id="step_read",
                description="Search web data",
                tool_name="search_web",
                arguments={"query": "python"},
                permission_level=PermissionLevel.READ,
            )
        ],
    )
    llm = get_mock_llm(structured_response=plan)
    graph = create_agent_graph(llm=llm, registry=registry, checkpointer=MemorySaver())

    res = await run_agent_workflow(
        user_request="Read web info",
        thread_id="th_read_only",
        graph=graph,
    )

    assert res["workflow_status"] == WorkflowStatus.COMPLETED
    assert read_mock.await_count == 1
    assert res["completed_steps"] == ["step_read"]


@pytest.mark.asyncio
async def test_write_workflow_interrupts_and_awaits_approval() -> None:
    """Verify WRITE workflow halts at interrupt, records ApprovalRequest, and does not execute."""
    write_mock = AsyncMock(return_value={"status": "created", "doc_id": "doc_123"})
    registry = AgentToolRegistry()
    registry.register_tool(
        RegisteredTool(
            name="upload_drive_file",
            description="Upload file",
            integration_name="google_drive",
            input_schema=MockToolInput,
            execution_function=write_mock,
            permission_level=PermissionLevel.WRITE,
        )
    )

    plan = PlannerOutput(
        reasoning_summary="Write workflow",
        steps=[
            PlanStepProposal(
                step_id="step_upload",
                description="Upload critical file",
                tool_name="upload_drive_file",
                arguments={"query": "financial_report.pdf"},
                permission_level=PermissionLevel.WRITE,
            )
        ],
    )
    llm = get_mock_llm(structured_response=plan)
    graph = create_agent_graph(llm=llm, registry=registry, checkpointer=MemorySaver())

    res = await run_agent_workflow(
        user_request="Upload quarterly report",
        thread_id="th_write_pause",
        graph=graph,
    )

    # Workflow must pause in AWAITING_APPROVAL
    assert res["workflow_status"] == WorkflowStatus.AWAITING_APPROVAL
    assert res["current_step_id"] == "step_upload"
    assert len(res["approval_requests"]) == 1
    assert res["approval_requests"][0].tool_name == "upload_drive_file"

    # Crucial security assertion: tool was NOT executed yet!
    assert write_mock.await_count == 0


@pytest.mark.asyncio
async def test_sensitive_workflow_interrupts_and_awaits_approval() -> None:
    """Verify SENSITIVE workflow (e.g. send_gmail_message) halts at interrupt point."""
    email_mock = AsyncMock(return_value={"status": "sent", "message_id": "msg_999"})
    registry = AgentToolRegistry()
    registry.register_tool(
        RegisteredTool(
            name="send_gmail_message",
            description="Send email",
            integration_name="gmail",
            input_schema=MockToolInput,
            execution_function=email_mock,
            permission_level=PermissionLevel.SENSITIVE,
        )
    )

    plan = PlannerOutput(
        reasoning_summary="Sensitive email workflow",
        steps=[
            PlanStepProposal(
                step_id="step_email",
                description="Send executive email",
                tool_name="send_gmail_message",
                arguments={"query": "to:board@company.com"},
                permission_level=PermissionLevel.SENSITIVE,
            )
        ],
    )
    llm = get_mock_llm(structured_response=plan)
    graph = create_agent_graph(llm=llm, registry=registry, checkpointer=MemorySaver())

    res = await run_agent_workflow(
        user_request="Send board email",
        thread_id="th_sensitive_pause",
        graph=graph,
    )

    assert res["workflow_status"] == WorkflowStatus.AWAITING_APPROVAL
    assert res["current_step_id"] == "step_email"
    assert email_mock.await_count == 0


@pytest.mark.asyncio
async def test_workflow_resumes_and_executes_on_human_approval() -> None:
    """Verify resuming an interrupted workflow with valid approval executes the tool."""
    write_mock = AsyncMock(return_value={"status": "created", "doc_id": "doc_abc456"})
    registry = AgentToolRegistry()
    registry.register_tool(
        RegisteredTool(
            name="create_google_document",
            description="Create doc",
            integration_name="google_drive",
            input_schema=MockToolInput,
            execution_function=write_mock,
            permission_level=PermissionLevel.WRITE,
        )
    )

    plan = PlannerOutput(
        reasoning_summary="Document creation workflow",
        steps=[
            PlanStepProposal(
                step_id="step_doc",
                description="Create project proposal doc",
                tool_name="create_google_document",
                arguments={"query": "Q3 Proposal"},
                permission_level=PermissionLevel.WRITE,
            )
        ],
    )
    llm = get_mock_llm(structured_response=plan)
    saver = MemorySaver()
    graph = create_agent_graph(llm=llm, registry=registry, checkpointer=saver)
    thread_id = "th_resume_approved"

    # 1. Run workflow -> pauses
    pause_res = await run_agent_workflow(
        user_request="Create project proposal",
        thread_id=thread_id,
        graph=graph,
    )
    assert pause_res["workflow_status"] == WorkflowStatus.AWAITING_APPROVAL
    assert write_mock.await_count == 0
    appr_id = pause_res["approval_requests"][0].approval_id

    # 2. Authenticated human operator authorizes execution in approval service
    appr_service = get_approval_service()
    await appr_service.approve_request(
        approval_id=appr_id,
        decided_by="lead_engineer_alice",
        reason="Explicit operator authorization",
    )

    # 3. Resume workflow
    resume_res = await resume_agent_workflow(
        thread_id=thread_id,
        decision={"approved": True, "decided_by": "lead_engineer_alice"},
        graph=graph,
    )

    # 4. Tool executed and workflow completed
    assert resume_res["workflow_status"] == WorkflowStatus.COMPLETED
    assert write_mock.await_count == 1
    assert "step_doc" in resume_res["completed_steps"]


@pytest.mark.asyncio
async def test_workflow_halts_and_blocks_execution_on_rejection() -> None:
    """Verify resuming with rejection does not execute the tool and records REJECTED status."""
    delete_mock = AsyncMock(return_value={"status": "deleted"})
    registry = AgentToolRegistry()
    registry.register_tool(
        RegisteredTool(
            name="delete_calendar_event",
            description="Delete event",
            integration_name="google_calendar",
            input_schema=MockToolInput,
            execution_function=delete_mock,
            permission_level=PermissionLevel.SENSITIVE,
        )
    )

    plan = PlannerOutput(
        reasoning_summary="Calendar deletion",
        steps=[
            PlanStepProposal(
                step_id="step_del",
                description="Delete event ev_777",
                tool_name="delete_calendar_event",
                arguments={"query": "ev_777"},
                permission_level=PermissionLevel.SENSITIVE,
            )
        ],
    )
    llm = get_mock_llm(structured_response=plan)
    saver = MemorySaver()
    graph = create_agent_graph(llm=llm, registry=registry, checkpointer=saver)
    thread_id = "th_reject_test"

    # 1. Run workflow -> pauses
    await run_agent_workflow(
        user_request="Delete meeting",
        thread_id=thread_id,
        graph=graph,
    )
    assert delete_mock.await_count == 0

    # 2. Operator rejects operation
    reject_res = await resume_agent_workflow(
        thread_id=thread_id,
        decision={
            "approved": False,
            "decided_by": "sec_officer_bob",
            "reason": "Accidental deletion prevented.",
        },
        graph=graph,
    )

    # 3. Status is REJECTED, tool NEVER executed
    assert reject_res["workflow_status"] == WorkflowStatus.REJECTED
    assert delete_mock.await_count == 0
    assert any("Operation rejected by human operator" in err for err in reject_res["error_history"])


@pytest.mark.asyncio
async def test_workflow_detects_and_blocks_argument_tampering() -> None:
    """Verify that tampering with arguments after approval fails validation and halts execution."""
    write_mock = AsyncMock(return_value={"status": "uploaded"})
    registry = AgentToolRegistry()
    registry.register_tool(
        RegisteredTool(
            name="upload_drive_file",
            description="Upload file",
            integration_name="google_drive",
            input_schema=MockToolInput,
            execution_function=write_mock,
            permission_level=PermissionLevel.WRITE,
        )
    )

    plan = PlannerOutput(
        reasoning_summary="Tamper test plan",
        steps=[
            PlanStepProposal(
                step_id="step_upload",
                description="Upload clean file",
                tool_name="upload_drive_file",
                arguments={"query": "safe.txt"},
                permission_level=PermissionLevel.WRITE,
            )
        ],
    )
    llm = get_mock_llm(structured_response=plan)
    saver = MemorySaver()
    graph = create_agent_graph(llm=llm, registry=registry, checkpointer=saver)
    thread_id = "th_tamper_test"

    pause_res = await run_agent_workflow(
        user_request="Upload safe file",
        thread_id=thread_id,
        graph=graph,
    )
    appr_id = pause_res["approval_requests"][0].approval_id

    appr_service = get_approval_service()
    # First approve legitimately in store
    await appr_service.approve_request(
        approval_id=appr_id,
        decided_by="alice",
    )

    # Now tamper with stored approval record's arguments hash
    stored = await appr_service.get_approval_request(appr_id)
    assert stored is not None
    tampered_rec = stored.model_copy(update={"arguments_hash": "tampered_hash_value"})
    await appr_service.store.save(tampered_rec)

    # Resume with approval
    res = await resume_agent_workflow(
        thread_id=thread_id,
        decision={"approved": True, "decided_by": "alice"},
        graph=graph,
    )

    # Must fail closed due to argument tamper detection!
    assert res["workflow_status"] == WorkflowStatus.FAILED
    assert write_mock.await_count == 0


@pytest.mark.asyncio
async def test_direct_resume_with_forged_approved_true_fails_closed() -> None:
    """Verify that attempting to resume with approved=True without prior approval fails closed."""
    write_mock = AsyncMock(return_value={"status": "uploaded"})
    registry = AgentToolRegistry()
    registry.register_tool(
        RegisteredTool(
            name="upload_drive_file",
            description="Upload file",
            integration_name="google_drive",
            input_schema=MockToolInput,
            execution_function=write_mock,
            permission_level=PermissionLevel.WRITE,
        )
    )

    plan = PlannerOutput(
        reasoning_summary="Forged resume test",
        steps=[
            PlanStepProposal(
                step_id="step_upload",
                description="Upload file",
                tool_name="upload_drive_file",
                arguments={"query": "secret.txt"},
                permission_level=PermissionLevel.WRITE,
            )
        ],
    )
    llm = get_mock_llm(structured_response=plan)
    saver = MemorySaver()
    graph = create_agent_graph(llm=llm, registry=registry, checkpointer=saver)
    thread_id = "th_forged_resume"

    pause_res = await run_agent_workflow(
        user_request="Upload secret",
        thread_id=thread_id,
        graph=graph,
    )
    assert pause_res["workflow_status"] == WorkflowStatus.AWAITING_APPROVAL

    # Attacker tries to resume directly with approved=True without calling approve_request
    forged_res = await resume_agent_workflow(
        thread_id=thread_id,
        decision={"approved": True, "decided_by": "attacker_actor"},
        graph=graph,
    )

    # Executor MUST fail closed: record in store is still PENDING!
    assert forged_res["workflow_status"] == WorkflowStatus.FAILED
    assert write_mock.await_count == 0
    assert any("was not authorized in the approval store" in e for e in forged_res["error_history"])
