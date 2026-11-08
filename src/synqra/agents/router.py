"""Tool Router Node and Dependency Resolver for SYNQRA LangGraph Engine.

Evaluates workflow progress against the ExecutionPlan, resolves dependencies,
and schedules runnable tasks with bounded concurrency for independent READ operations
while strictly isolating sequential WRITE operations.
"""

import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import RunnableConfig

from synqra.agents.exceptions import SynqraAgentError
from synqra.agents.schemas import ExecutionPlan, PermissionLevel, PlanStep, WorkflowStatus
from synqra.agents.state import AgentState
from synqra.agents.tool_registry import AgentToolRegistry
from synqra.agents.tools import get_default_agent_tool_registry

logger = logging.getLogger("synqra.agents.router")

DEFAULT_MAX_PARALLEL_READS: int = 4


@dataclass
class ScheduledBatch:
    """Represents a scheduled wave of steps ready for execution."""

    step_ids: list[str]
    is_parallel: bool
    permission_level: PermissionLevel


def route_runnable_steps(
    plan: ExecutionPlan,
    completed_steps: list[str],
    pending_steps: list[str],
    registry: AgentToolRegistry | None = None,
    max_parallel: int = DEFAULT_MAX_PARALLEL_READS,
) -> ScheduledBatch | None:
    """Determine the next batch of steps eligible for execution based on completed dependencies.

    Args:
        plan: Active ExecutionPlan.
        completed_steps: Set/list of step_ids already completed.
        pending_steps: List of remaining step_ids in topological order.
        registry: AgentToolRegistry for validation.
        max_parallel: Max concurrent independent READ operations.

    Returns:
        ScheduledBatch containing runnable step IDs, or None if no steps are currently runnable.

    Raises:
        SynqraAgentError: If a scheduled step references an unknown or disabled tool.
    """
    if not pending_steps:
        return None

    tool_reg = registry or get_default_agent_tool_registry()
    completed_set = set(completed_steps)

    # 1. Identify all pending steps whose prerequisites have completed
    runnable_steps: list[PlanStep] = []
    for step_id in pending_steps:
        step = plan.get_step(step_id)
        if step is None:
            continue
        if set(step.dependencies).issubset(completed_set):
            # Validate that tool exists and is active in registry
            tool = tool_reg.get_tool(step.tool_name)
            if tool is None:
                raise SynqraAgentError(
                    f"Step '{step.step_id}' references unknown tool: '{step.tool_name}'."
                )
            if not tool.is_active:
                raise SynqraAgentError(
                    f"Step '{step.step_id}' references disabled tool: '{step.tool_name}'."
                )
            runnable_steps.append(step)

    if not runnable_steps:
        return None

    # 2. Safety Rule: If any runnable step is WRITE or SENSITIVE, run it singularly (isolated)
    first_step = runnable_steps[0]
    if first_step.permission_level in (PermissionLevel.WRITE, PermissionLevel.SENSITIVE):
        return ScheduledBatch(
            step_ids=[first_step.step_id],
            is_parallel=False,
            permission_level=first_step.permission_level,
        )

    # 3. For READ-only steps: collect consecutive independent READ steps up to max_parallel
    read_batch: list[str] = []
    for step in runnable_steps:
        if step.permission_level == PermissionLevel.READ:
            read_batch.append(step.step_id)
            if len(read_batch) >= max_parallel:
                break
        else:
            # Do not mix WRITE steps with parallel READ batches
            break

    return ScheduledBatch(
        step_ids=read_batch,
        is_parallel=len(read_batch) > 1,
        permission_level=PermissionLevel.READ,
    )


async def router_node(
    state: AgentState | dict[str, Any],
    config: RunnableConfig | None = None,
    registry: AgentToolRegistry | None = None,
    max_parallel: int = DEFAULT_MAX_PARALLEL_READS,
) -> dict[str, Any]:
    """LangGraph node scheduling the next runnable batch of execution steps."""
    _ = config
    if isinstance(state, dict):
        plan_raw = state.get("execution_plan")
        plan = ExecutionPlan.model_validate(plan_raw) if isinstance(plan_raw, dict) else plan_raw
        completed = list(state.get("completed_steps", []))
        pending = list(state.get("pending_steps", []))
        curr_status = state.get("workflow_status", WorkflowStatus.EXECUTING)
    else:
        plan = state.execution_plan
        completed = list(state.completed_steps)
        pending = list(state.pending_steps)
        curr_status = state.workflow_status

    if not plan or not pending:
        return {
            "current_step_id": None,
            "workflow_status": WorkflowStatus.COMPLETED,
        }

    try:
        batch = route_runnable_steps(
            plan=plan,
            completed_steps=completed,
            pending_steps=pending,
            registry=registry,
            max_parallel=max_parallel,
        )
    except Exception as exc:
        logger.error("Router resolution failed: %s", exc)
        curr_errors = (
            state.get("error_history", []) if isinstance(state, dict) else list(state.error_history)
        )
        return {
            "workflow_status": WorkflowStatus.FAILED,
            "error_history": curr_errors + [f"Routing error: {exc}"],
        }

    if not batch or not batch.step_ids:
        # No steps currently runnable (either deadlock or waiting)
        logger.warning("No steps eligible to run despite pending steps: %s", pending)
        return {
            "current_step_id": None,
            "workflow_status": WorkflowStatus.FAILED if pending else WorkflowStatus.COMPLETED,
        }

    primary_step_id = batch.step_ids[0]
    return {
        "current_step_id": primary_step_id,
        "workflow_status": curr_status,
    }
