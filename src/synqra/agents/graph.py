"""LangGraph Orchestration Engine for SYNQRA.

Assembles the full multi-agent state graph connecting Planner, Tool Router,
Executor, Evaluator, Error Recovery, and Result Aggregator nodes with stateful
checkpointing, bounded concurrency, and strict permission safety boundaries.
"""

import logging
import uuid
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from synqra.agents.aggregator import aggregator_node
from synqra.agents.checkpointer import get_checkpointer
from synqra.agents.evaluator import (
    DEFAULT_MAX_RETRIES,
    EvaluatorAction,
    evaluate_workflow_state,
    evaluator_node,
)
from synqra.agents.executor import DEFAULT_EXECUTOR_CONCURRENCY, executor_node
from synqra.agents.llm import get_llm
from synqra.agents.planner import DEFAULT_MAX_PLAN_STEPS, planner_node
from synqra.agents.recovery import recovery_node
from synqra.agents.router import router_node
from synqra.agents.schemas import ApprovalRequest, PermissionLevel, WorkflowStatus
from synqra.agents.state import AgentState
from synqra.agents.tool_adapter import AgentToolAdapter
from synqra.agents.tool_registry import AgentToolRegistry
from synqra.agents.tools import get_default_agent_tool_registry
from synqra.core.config import get_settings

logger = logging.getLogger("synqra.agents.graph")


def _planner_condition(state: AgentState | dict[str, Any]) -> str:
    """Evaluate plan validity after planner node."""
    status = state.get("workflow_status") if isinstance(state, dict) else state.workflow_status
    plan = state.get("execution_plan") if isinstance(state, dict) else state.execution_plan

    if status == WorkflowStatus.FAILED or not plan or not getattr(plan, "steps", []):
        return "aggregator"
    return "router"


def _router_condition(state: AgentState | dict[str, Any]) -> str:
    """Route after scheduling step(s) in router node."""
    status = state.get("workflow_status") if isinstance(state, dict) else state.workflow_status
    curr_step = state.get("current_step_id") if isinstance(state, dict) else state.current_step_id

    if status == WorkflowStatus.AWAITING_APPROVAL:
        # Pause execution: human approval required
        return "end"

    if status == WorkflowStatus.FAILED or not curr_step:
        return "aggregator"

    return "executor"


def _evaluator_condition(state: AgentState | dict[str, Any]) -> str:
    """Determine next action based on evaluator node decision."""
    status = state.get("workflow_status") if isinstance(state, dict) else state.workflow_status
    if status in (WorkflowStatus.REJECTED, WorkflowStatus.CANCELLED):
        return "aggregator"

    eval_res = evaluate_workflow_state(state=state)

    if eval_res.action == EvaluatorAction.AWAITING_APPROVAL:
        return "end"
    if eval_res.action == EvaluatorAction.RETRY:
        return "recovery"
    if eval_res.action == EvaluatorAction.CONTINUE:
        return "router"
    # COMPLETE or FAIL routes to aggregator for final synthesis
    return "aggregator"


def _recovery_condition(state: AgentState | dict[str, Any]) -> str:
    """Route after error recovery attempt."""
    status = state.get("workflow_status") if isinstance(state, dict) else state.workflow_status
    pending = state.get("pending_steps", []) if isinstance(state, dict) else state.pending_steps

    if status == WorkflowStatus.EXECUTING and pending:
        return "router"
    return "aggregator"


def create_agent_graph(
    llm: BaseChatModel | None = None,
    registry: AgentToolRegistry | None = None,
    adapter: AgentToolAdapter | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    max_steps: int = DEFAULT_MAX_PLAN_STEPS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    max_concurrency: int = DEFAULT_EXECUTOR_CONCURRENCY,
) -> CompiledStateGraph:
    """Construct and compile the complete SYNQRA LangGraph agentic workflow.

    Args:
        llm: Optional ChatModel instance for planner and aggregator.
        registry: AgentToolRegistry containing tools.
        adapter: Tool adapter for executing integration functions.
        checkpointer: State persistence checkpointer (defaults to in-memory MemorySaver).
        max_steps: Maximum steps permitted in an execution plan.
        max_retries: Maximum bounded retry count for transient failures.
        max_concurrency: Maximum concurrent independent READ operations.

    Returns:
        CompiledStateGraph ready for execution with state and checkpoint support.
    """
    tool_reg = registry or get_default_agent_tool_registry()
    active_llm = llm or get_llm()
    saver = MemorySaver() if checkpointer is None else checkpointer

    builder = StateGraph(AgentState)

    # 1. Define Nodes
    async def _planner(
        state: AgentState | dict[str, Any], config: RunnableConfig | None = None
    ) -> dict[str, Any]:
        return await planner_node(
            state=state,
            config=config,
            llm=active_llm,
            registry=tool_reg,
            max_steps=max_steps,
        )

    async def _router(
        state: AgentState | dict[str, Any], config: RunnableConfig | None = None
    ) -> dict[str, Any]:
        return await router_node(
            state=state,
            config=config,
            registry=tool_reg,
            max_parallel=max_concurrency,
        )

    async def _executor(
        state: AgentState | dict[str, Any], config: RunnableConfig | None = None
    ) -> dict[str, Any]:
        return await executor_node(
            state=state,
            config=config,
            registry=tool_reg,
            adapter=adapter,
            max_concurrency=max_concurrency,
        )

    async def _evaluator(
        state: AgentState | dict[str, Any], config: RunnableConfig | None = None
    ) -> dict[str, Any]:
        return await evaluator_node(
            state=state,
            config=config,
            max_retries=max_retries,
        )

    async def _recovery(
        state: AgentState | dict[str, Any], config: RunnableConfig | None = None
    ) -> dict[str, Any]:
        return await recovery_node(
            state=state,
            config=config,
            max_retries=max_retries,
        )

    async def _aggregator(
        state: AgentState | dict[str, Any], config: RunnableConfig | None = None
    ) -> dict[str, Any]:
        return await aggregator_node(
            state=state,
            config=config,
            llm=active_llm,
        )

    builder.add_node("planner", _planner)
    builder.add_node("router", _router)
    builder.add_node("executor", _executor)
    builder.add_node("evaluator", _evaluator)
    builder.add_node("recovery", _recovery)
    builder.add_node("aggregator", _aggregator)

    # 2. Wire Graph Edges
    builder.add_edge(START, "planner")

    builder.add_conditional_edges(
        "planner",
        _planner_condition,
        {
            "router": "router",
            "aggregator": "aggregator",
        },
    )

    builder.add_conditional_edges(
        "router",
        _router_condition,
        {
            "executor": "executor",
            "aggregator": "aggregator",
            "end": END,
        },
    )

    builder.add_edge("executor", "evaluator")

    builder.add_conditional_edges(
        "evaluator",
        _evaluator_condition,
        {
            "router": "router",
            "recovery": "recovery",
            "aggregator": "aggregator",
            "end": END,
        },
    )

    builder.add_conditional_edges(
        "recovery",
        _recovery_condition,
        {
            "router": "router",
            "aggregator": "aggregator",
        },
    )

    builder.add_edge("aggregator", END)

    # 3. Compile Graph with Checkpointer
    return builder.compile(checkpointer=saver)


async def run_agent_workflow(
    user_request: str,
    thread_id: str | None = None,
    graph: CompiledStateGraph | None = None,
    llm: BaseChatModel | None = None,
    registry: AgentToolRegistry | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> dict[str, Any]:
    """Execute an agent workflow end-to-end starting with a user request.

    Args:
        user_request: Natural-language instructions for the agent.
        thread_id: Optional thread ID for persistent checkpoints.
        graph: Optional compiled graph instance.
        llm: Optional ChatModel instance.
        registry: Optional AgentToolRegistry.
        checkpointer: Optional explicit state persistence checkpointer.

    Returns:
        Final state dictionary upon workflow completion or interruption.
    """
    if graph is not None:
        app = graph
    else:
        active_cp = checkpointer
        if active_cp is None and get_settings().checkpointer_type == "postgres":
            active_cp = await get_checkpointer()
        app = create_agent_graph(llm=llm, registry=registry, checkpointer=active_cp)
    tid = thread_id or f"th_{uuid.uuid4().hex[:12]}"
    wid = f"wf_{uuid.uuid4().hex[:12]}"

    initial_input = {
        "workflow_id": wid,
        "user_request": user_request,
        "thread_id": tid,
        "workflow_status": WorkflowStatus.PENDING,
    }

    config = {"configurable": {"thread_id": tid}}
    result = await app.ainvoke(initial_input, config=config)

    # Enrich state if execution paused on human approval interrupt
    if "__interrupt__" in result and result["__interrupt__"]:
        int_obj = result["__interrupt__"][0]
        int_val = int_obj.value if hasattr(int_obj, "value") else int_obj
        step_id = int_val.get("step_id")
        appr_id = int_val.get("approval_id")
        tool_name = int_val.get("tool_name")
        args = int_val.get("arguments", {})
        perm = int_val.get("permission_level", "sensitive")
        desc = int_val.get("description", "")
        wf_id = int_val.get("workflow_id", result.get("workflow_id", ""))

        appr_req = ApprovalRequest(
            approval_id=appr_id,
            workflow_id=wf_id,
            tool_name=tool_name,
            arguments=args,
            reason=desc,
            permission_level=PermissionLevel(perm),
        )

        result["workflow_status"] = WorkflowStatus.AWAITING_APPROVAL
        result["current_step_id"] = step_id
        current_reqs = list(result.get("approval_requests", []))
        if not any(r.approval_id == appr_id for r in current_reqs):
            result["approval_requests"] = current_reqs + [appr_req]

    return result


async def resume_agent_workflow(
    thread_id: str,
    decision: dict[str, Any] | Any,
    graph: CompiledStateGraph | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> dict[str, Any]:
    """Resume an interrupted agent workflow with an operator's approval decision.

    Args:
        thread_id: Thread ID of the paused workflow.
        decision: Approval decision dictionary or Pydantic model.
        graph: Optional compiled graph instance.
        checkpointer: Optional explicit state persistence checkpointer.

    Returns:
        Workflow state dictionary after executing or rejecting.
    """
    if graph is not None:
        app = graph
    else:
        active_cp = checkpointer
        if active_cp is None and get_settings().checkpointer_type == "postgres":
            active_cp = await get_checkpointer()
        app = create_agent_graph(checkpointer=active_cp)
    decision_dict = decision.model_dump() if hasattr(decision, "model_dump") else dict(decision)

    config = {"configurable": {"thread_id": thread_id}}
    result = await app.ainvoke(Command(resume=decision_dict), config=config)

    # Check if a subsequent step triggered another human approval interruption
    if "__interrupt__" in result and result["__interrupt__"]:
        int_obj = result["__interrupt__"][0]
        int_val = int_obj.value if hasattr(int_obj, "value") else int_obj
        step_id = int_val.get("step_id")
        appr_id = int_val.get("approval_id")
        tool_name = int_val.get("tool_name")
        args = int_val.get("arguments", {})
        perm = int_val.get("permission_level", "sensitive")
        desc = int_val.get("description", "")
        wf_id = int_val.get("workflow_id", result.get("workflow_id", ""))

        appr_req = ApprovalRequest(
            approval_id=appr_id,
            workflow_id=wf_id,
            tool_name=tool_name,
            arguments=args,
            reason=desc,
            permission_level=PermissionLevel(perm),
        )

        result["workflow_status"] = WorkflowStatus.AWAITING_APPROVAL
        result["current_step_id"] = step_id
        current_reqs = list(result.get("approval_requests", []))
        if not any(r.approval_id == appr_id for r in current_reqs):
            result["approval_requests"] = current_reqs + [appr_req]

    return result
