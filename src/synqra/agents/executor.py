"""Executor Node for SYNQRA LangGraph Agentic Engine.

Executes scheduled tool steps exclusively through the Phase 9.2 Tool Bridge,
failing closed if human approval is required for WRITE/SENSITIVE operations,
and executing independent READ steps with bounded concurrency.
"""

import asyncio
import contextlib
import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from synqra.agents.approval.service import get_approval_service
from synqra.agents.permissions import ApprovalContext
from synqra.agents.router import route_runnable_steps
from synqra.agents.schemas import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStatus,
    ExecutionPlan,
    PermissionLevel,
    PlanStep,
    StepStatus,
    ToolExecutionResult,
    WorkflowStatus,
)
from synqra.agents.state import AgentState
from synqra.agents.tool_adapter import AgentToolAdapter
from synqra.agents.tool_registry import AgentToolRegistry
from synqra.agents.tools import execute_tool, get_default_agent_tool_registry

logger = logging.getLogger("synqra.agents.executor")

DEFAULT_EXECUTOR_CONCURRENCY: int = 4


async def execute_plan_step(
    step: PlanStep,
    workflow_id: str,
    approval_context: ApprovalContext | None = None,
    registry: AgentToolRegistry | None = None,
    adapter: AgentToolAdapter | None = None,
    timeout: float = 60.0,
) -> ToolExecutionResult:
    """Execute a single PlanStep through the centralized Tool Bridge."""
    return await execute_tool(
        tool_name=step.tool_name,
        arguments=step.arguments,
        workflow_id=workflow_id,
        approval_context=approval_context,
        registry=registry,
        adapter=adapter,
        timeout=timeout,
    )


async def executor_node(
    state: AgentState | dict[str, Any],
    config: RunnableConfig | None = None,
    registry: AgentToolRegistry | None = None,
    adapter: AgentToolAdapter | None = None,
    max_concurrency: int = DEFAULT_EXECUTOR_CONCURRENCY,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """LangGraph node executing runnable tool steps with strict permission checking."""
    _ = config
    # 1. Normalize state input
    if isinstance(state, dict):
        workflow_id = state.get("workflow_id", "wf_default")
        thread_id = (
            state.get("thread_id")
            or (config.get("configurable", {}).get("thread_id") if config else None)
            or "th_default"
        )
        plan_raw = state.get("execution_plan")
        plan = ExecutionPlan.model_validate(plan_raw) if isinstance(plan_raw, dict) else plan_raw
        completed = list(state.get("completed_steps", []))
        pending = list(state.get("pending_steps", []))
        results = dict(state.get("execution_results", {}))
        approval_reqs = list(state.get("approval_requests", []))
        errors = list(state.get("error_history", []))
    else:
        workflow_id = state.workflow_id
        thread_id = state.thread_id
        plan = state.execution_plan
        completed = list(state.completed_steps)
        pending = list(state.pending_steps)
        results = dict(state.execution_results)
        approval_reqs = list(state.approval_requests)
        errors = list(state.error_history)

    if not plan or not pending:
        return {
            "current_step_id": None,
            "workflow_status": WorkflowStatus.COMPLETED,
        }

    tool_reg = registry or get_default_agent_tool_registry()

    # 2. Determine currently runnable batch via router logic
    batch = route_runnable_steps(
        plan=plan,
        completed_steps=completed,
        pending_steps=pending,
        registry=tool_reg,
        max_parallel=max_concurrency,
    )

    if not batch or not batch.step_ids:
        return {
            "current_step_id": None,
            "workflow_status": WorkflowStatus.FAILED if pending else WorkflowStatus.COMPLETED,
        }

    # 3. Check for approval requirement on WRITE or SENSITIVE operations
    raw_steps: list[PlanStep] = [plan.get_step(sid) for sid in batch.step_ids if plan.get_step(sid)]

    # Filter out steps that have already been executed with a persisted success result
    steps_to_run: list[PlanStep] = []
    for s in raw_steps:
        if s.step_id in completed or (s.step_id in results and results[s.step_id].success):
            logger.info(
                "Step '%s' (%s) already has a persisted successful execution result; "
                "reusing without re-execution.",
                s.step_id,
                s.tool_name,
            )
            s.transition_status(StepStatus.COMPLETED)
            if s.step_id in pending:
                pending.remove(s.step_id)
            if s.step_id not in completed:
                completed.append(s.step_id)
        else:
            steps_to_run.append(s)

    if not steps_to_run:
        # All runnable steps in this batch were already executed and persisted
        return {
            "execution_plan": plan,
            "completed_steps": completed,
            "pending_steps": pending,
            "execution_results": results,
            "current_step_id": None,
            "workflow_status": WorkflowStatus.COMPLETED
            if not pending
            else WorkflowStatus.EXECUTING,
        }

    approved_steps: list[PlanStep] = []
    step_contexts: dict[str, ApprovalContext] = {}
    appr_service = get_approval_service()

    for step in steps_to_run:
        if step.permission_level in (PermissionLevel.WRITE, PermissionLevel.SENSITIVE):
            # Check if an approval record was already generated for this step
            existing_rec = await appr_service.find_request_for_step(
                workflow_id=workflow_id,
                step_id=step.step_id,
                thread_id=thread_id,
                tool_name=step.tool_name,
            )
            if existing_rec:
                rec = existing_rec
            else:
                rec = await appr_service.create_approval_request(
                    workflow_id=workflow_id,
                    thread_id=thread_id,
                    step_id=step.step_id,
                    tool_name=step.tool_name,
                    arguments=step.arguments,
                    permission_level=step.permission_level,
                    description=step.description,
                )
            appr_request = ApprovalRequest(
                approval_id=rec.approval_id,
                workflow_id=workflow_id,
                tool_name=step.tool_name,
                arguments=step.arguments,
                reason=rec.description,
                permission_level=step.permission_level,
                created_at=rec.created_at,
                expires_at=rec.expires_at,
                status=rec.status,
            )
            step.transition_status(StepStatus.AWAITING_APPROVAL)
            logger.info(
                "Execution paused for human approval on step '%s' (tool: %s)",
                step.step_id,
                step.tool_name,
            )

            interrupt_payload = {
                "approval_id": rec.approval_id,
                "workflow_id": workflow_id,
                "thread_id": thread_id,
                "step_id": step.step_id,
                "tool_name": step.tool_name,
                "permission_level": step.permission_level.value,
                "description": rec.description,
                "arguments": step.arguments,
                "arguments_hash": rec.arguments_hash,
                "expires_at": rec.expires_at.isoformat(),
                "status": rec.status.value,
            }

            try:
                resume_data = interrupt(interrupt_payload)
            except RuntimeError:
                # Outside a LangGraph runnable context (e.g. direct isolated node tests)
                return {
                    "current_step_id": step.step_id,
                    "workflow_status": WorkflowStatus.AWAITING_APPROVAL,
                    "approval_requests": approval_reqs + [appr_request],
                }

            # If execution resumes here, evaluate human decision
            res_dict = (
                resume_data
                if isinstance(resume_data, dict)
                else (resume_data.model_dump() if hasattr(resume_data, "model_dump") else {})
            )
            is_approved = res_dict.get("approved", False)
            decider = res_dict.get("decided_by", "human_operator")
            reason = res_dict.get("reason")

            if not is_approved:
                step.transition_status(StepStatus.REJECTED)
                step.error = (
                    f"Operation rejected by human operator '{decider}': "
                    f"{reason or 'No reason provided.'}"
                )
                logger.warning(step.error)
                if rec.status == ApprovalStatus.PENDING:
                    with contextlib.suppress(Exception):
                        await appr_service.reject_request(
                            rec.approval_id, decided_by=decider, reason=reason
                        )
                return {
                    "execution_plan": plan,
                    "current_step_id": step.step_id,
                    "workflow_status": WorkflowStatus.REJECTED,
                    "error_history": errors + [step.error],
                }

            # Verify that a legitimate, unexpired APPROVED record exists in the approval store.
            # Arbitrary resume payloads with approved=True without prior approval are rejected.
            current_rec = await appr_service.get_approval_request(rec.approval_id)

            # Check if arguments were tampered with post-approval
            if current_rec and current_rec.arguments_hash != rec.arguments_hash:
                step.transition_status(StepStatus.FAILED)
                step.error = (
                    f"Execution blocked: step arguments for '{step.step_id}' were modified "
                    f"after approval was generated. Approved hash '{current_rec.arguments_hash}' "
                    f"does not match current hash '{rec.arguments_hash}'."
                )
                logger.error(step.error)
                return {
                    "execution_plan": plan,
                    "current_step_id": step.step_id,
                    "workflow_status": WorkflowStatus.FAILED,
                    "error_history": errors + [step.error],
                }

            # Handle already consumed approval
            if current_rec and current_rec.status == ApprovalStatus.CONSUMED:
                if step.step_id in results and results[step.step_id].success:
                    logger.info(
                        "Step '%s' approval '%s' was already consumed and execution result "
                        "is persisted; reusing persisted result.",
                        step.step_id,
                        rec.approval_id,
                    )
                    continue
                # Approval consumed but no success result persisted: ambiguous execution outcome!
                step.transition_status(StepStatus.FAILED)
                step.error = (
                    f"Ambiguous execution outcome on step '{step.step_id}': approval request "
                    f"'{rec.approval_id}' was already CONSUMED, but no completed execution "
                    "result was persisted. To prevent duplicate external side-effects, "
                    "ambiguous WRITE/SENSITIVE operations cannot be automatically re-executed."
                )
                logger.error(step.error)
                return {
                    "execution_plan": plan,
                    "current_step_id": step.step_id,
                    "workflow_status": WorkflowStatus.FAILED,
                    "error_history": errors + [step.error],
                }

            if not current_rec or current_rec.status != ApprovalStatus.APPROVED:
                is_rejected = current_rec and current_rec.status == ApprovalStatus.REJECTED
                is_expired = current_rec and current_rec.status == ApprovalStatus.EXPIRED
                step.transition_status(StepStatus.REJECTED if is_rejected else StepStatus.FAILED)
                step.error = (
                    f"Execution blocked: approval request '{rec.approval_id}' was not "
                    f"authorized in the approval store (status: "
                    f"'{current_rec.status.value if current_rec else 'missing'}'). "
                    f"A valid APPROVED record must exist prior to resumption; "
                    f"arbitrary resume with approved=True is strictly rejected."
                )
                logger.error(step.error)
                return {
                    "execution_plan": plan,
                    "current_step_id": step.step_id,
                    "workflow_status": (
                        WorkflowStatus.REJECTED
                        if is_rejected
                        else (WorkflowStatus.EXPIRED if is_expired else WorkflowStatus.FAILED)
                    ),
                    "error_history": errors + [step.error],
                }

            # Approved: consume single-use approval record
            consumed, consume_err = await appr_service.consume_approval(
                approval_id=rec.approval_id,
                workflow_id=workflow_id,
                thread_id=thread_id,
                step_id=step.step_id,
                tool_name=step.tool_name,
                arguments=step.arguments,
                permission_level=step.permission_level,
            )
            if not consumed:
                step.transition_status(StepStatus.FAILED)
                step.error = f"Approval authorization/consumption failed: {consume_err}"
                logger.error(step.error)
                return {
                    "execution_plan": plan,
                    "current_step_id": step.step_id,
                    "workflow_status": WorkflowStatus.FAILED,
                    "error_history": errors + [step.error],
                }

            decider_name = (
                current_rec.decision.decided_by if current_rec and current_rec.decision else decider
            )
            appr_decision = ApprovalDecision(
                approval_id=rec.approval_id,
                approved=True,
                decided_by=decider_name,
                reason=reason,
            )
            step_contexts[step.step_id] = ApprovalContext(
                approval_request=appr_request,
                approval_decision=appr_decision,
            )
            approved_steps.append(step)
        else:
            approved_steps.append(step)

    # 4. Execute the authorized steps (concurrently for READ, sequentially for single WRITE)
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _run_single(step_obj: PlanStep) -> tuple[str, ToolExecutionResult]:
        async with semaphore:
            step_obj.transition_status(StepStatus.RUNNING)
            exec_res = await execute_plan_step(
                step=step_obj,
                workflow_id=workflow_id,
                approval_context=step_contexts.get(step_obj.step_id),
                registry=tool_reg,
                adapter=adapter,
                timeout=timeout,
            )
            return step_obj.step_id, exec_res

    run_tasks = [_run_single(s) for s in approved_steps]
    step_results = await asyncio.gather(*run_tasks, return_exceptions=False)

    # 5. Process results and update tracking
    new_completed = list(completed)
    new_pending = list(pending)
    new_results = dict(results)
    new_errors = list(errors)

    has_failure = False
    for step_id, res in step_results:
        step_obj = plan.get_step(step_id)
        if step_obj is None:
            continue

        step_obj.execution_result = res
        new_results[step_id] = res

        if res.success:
            step_obj.transition_status(StepStatus.COMPLETED)
            if step_id in new_pending:
                new_pending.remove(step_id)
            if step_id not in new_completed:
                new_completed.append(step_id)
        else:
            step_obj.transition_status(StepStatus.FAILED)
            step_obj.error = res.error
            new_errors.append(f"Step '{step_id}' ({step_obj.tool_name}) failed: {res.error}")
            has_failure = True

    next_status = (
        WorkflowStatus.FAILED
        if has_failure
        else (WorkflowStatus.COMPLETED if not new_pending else WorkflowStatus.EXECUTING)
    )

    return {
        "execution_plan": plan,
        "completed_steps": new_completed,
        "pending_steps": new_pending,
        "execution_results": new_results,
        "error_history": new_errors,
        "current_step_id": None,
        "workflow_status": next_status,
    }
