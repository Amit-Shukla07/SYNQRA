"""Planner Node for SYNQRA LangGraph Agentic Engine.

Analyzes natural-language user requests, discovers available tools from AgentToolRegistry,
and constructs structured, dependency-validated ExecutionPlans via LLM structured outputs.
"""

import logging
from collections.abc import Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from synqra.agents.exceptions import InvalidExecutionPlanError
from synqra.agents.llm import get_llm
from synqra.agents.permissions import get_tool_permission_level
from synqra.agents.schemas import ExecutionPlan, PermissionLevel, PlanStep, WorkflowStatus
from synqra.agents.state import AgentState
from synqra.agents.tool_registry import AgentToolRegistry, RegisteredTool
from synqra.agents.tools import get_default_agent_tool_registry

logger = logging.getLogger("synqra.agents.planner")

DEFAULT_MAX_PLAN_STEPS: int = 10

PLANNER_SYSTEM_PROMPT = """You are the SYNQRA Orchestration Planner Agent.
Your role is to analyze the user's request and construct a concrete, multi-step execution plan
using ONLY the tools available in SYNQRA.

Follow these strict rules:
1. Break complex requests down into small, logical, dependency-ordered tasks.
2. Select tools ONLY from the provided Available Tools Catalog. Never invent tool names.
3. Every step must have a unique identifier (e.g. 'step_1', 'step_2').
4. If a step needs outputs from previous steps, list their step_ids in 'dependencies'.
5. Do NOT create circular dependencies or self-dependencies.
6. Generate valid arguments for each tool according to its parameter schema.
7. Classify permission_level accurately (read, write, sensitive).
8. The plan must not exceed {max_steps} steps.
9. You must NEVER execute tools or generate approval decisions yourself.

Available Tools Catalog:
{tools_catalog}
"""


class PlanStepProposal(BaseModel):
    """Structured proposal for a single task within an execution plan."""

    step_id: str = Field(description="Unique step identifier (e.g. 'step_1').")
    description: str = Field(description="Explanation of what this task accomplishes.")
    tool_name: str = Field(description="Exact name of the tool to be executed.")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value arguments for the tool conforming to its schema.",
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="List of step_ids that must complete before this step can run.",
    )
    permission_level: PermissionLevel = Field(
        default=PermissionLevel.READ,
        description="Required permission level for the tool (read, write, sensitive).",
    )


class PlannerOutput(BaseModel):
    """Structured response produced by the planner LLM."""

    reasoning_summary: str = Field(
        description="High-level summary of the planning logic and task decomposition."
    )
    steps: list[PlanStepProposal] = Field(
        default_factory=list,
        description="Ordered list of proposed plan steps.",
    )


def format_tools_catalog(tools: Sequence[RegisteredTool]) -> str:
    """Format registered tools and schemas into a clean description for the planner prompt."""
    lines: list[str] = []
    for tool in tools:
        if not tool.is_active:
            continue
        schema_props = {}
        if hasattr(tool.input_schema, "model_json_schema"):
            schema_props = tool.input_schema.model_json_schema().get("properties", {})
        param_list = ", ".join(schema_props.keys()) if schema_props else "none"
        lines.append(
            f"- **{tool.name}** ({tool.integration_name}, {tool.permission_level.value}):\n"
            f"  Description: {tool.description}\n"
            f"  Parameters: {param_list}"
        )
    return "\n".join(lines)


async def plan_workflow(
    user_request: str,
    llm: BaseChatModel | None = None,
    registry: AgentToolRegistry | None = None,
    max_steps: int = DEFAULT_MAX_PLAN_STEPS,
) -> ExecutionPlan:
    """Generate a structured, dependency-validated ExecutionPlan for a user request.

    Args:
        user_request: Natural-language user prompt.
        llm: Language model instance. Defaults to get_llm().
        registry: AgentToolRegistry instance. Defaults to global registry.
        max_steps: Maximum allowed number of steps in plan.

    Returns:
        Validated ExecutionPlan instance.

    Raises:
        InvalidExecutionPlanError: If planning fails or produces an invalid plan.
    """
    tool_reg = registry or get_default_agent_tool_registry()
    tools = tool_reg.list_tools(active_only=True)

    if not tools:
        raise InvalidExecutionPlanError("Cannot generate plan: no active tools in registry.")

    catalog_str = format_tools_catalog(tools)
    sys_prompt = PLANNER_SYSTEM_PROMPT.format(max_steps=max_steps, tools_catalog=catalog_str)

    active_llm = llm or get_llm()
    structured_planner = active_llm.with_structured_output(PlannerOutput)

    messages = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=user_request),
    ]

    try:
        raw_output = await structured_planner.ainvoke(messages)
    except Exception as exc:
        logger.error("LLM planning invocation failed: %s", exc)
        raise InvalidExecutionPlanError(f"Planner LLM failed to generate plan: {exc}") from exc

    if not isinstance(raw_output, PlannerOutput):
        if hasattr(raw_output, "model_dump"):
            raw_output = PlannerOutput.model_validate(raw_output.model_dump())
        elif isinstance(raw_output, dict):
            raw_output = PlannerOutput.model_validate(raw_output)
        else:
            raise InvalidExecutionPlanError(
                f"Unexpected planner output format: {type(raw_output).__name__}"
            )

    if len(raw_output.steps) > max_steps:
        raise InvalidExecutionPlanError(
            f"Generated plan exceeds maximum step limit: {len(raw_output.steps)} > {max_steps}."
        )

    # Convert proposals to validated PlanStep models and enforce tool allowlist
    validated_steps: list[PlanStep] = []
    for proposal in raw_output.steps:
        tool = tool_reg.get_tool(proposal.tool_name)
        if tool is None or not tool.is_active:
            raise InvalidExecutionPlanError(
                f"Plan references unknown or disabled tool: '{proposal.tool_name}'."
            )

        # Enforce canonical permission classification from catalog (anti-tampering)
        enforced_permission = get_tool_permission_level(proposal.tool_name)

        validated_steps.append(
            PlanStep(
                step_id=proposal.step_id,
                description=proposal.description,
                tool_name=proposal.tool_name,
                arguments=proposal.arguments,
                dependencies=proposal.dependencies,
                permission_level=enforced_permission,
            )
        )

    # Build and validate execution plan (validates uniqueness, dependencies, cycle detection)
    plan = ExecutionPlan(
        user_request=user_request,
        steps=validated_steps,
        reasoning_summary=raw_output.reasoning_summary,
        estimated_steps=len(validated_steps),
        status=WorkflowStatus.PLANNING,
    )

    return plan


async def planner_node(
    state: AgentState | dict[str, Any],
    config: RunnableConfig | None = None,
    llm: BaseChatModel | None = None,
    registry: AgentToolRegistry | None = None,
    max_steps: int = DEFAULT_MAX_PLAN_STEPS,
) -> dict[str, Any]:
    """LangGraph node function for request planning and decomposition."""
    _ = config
    user_req = state.get("user_request") if isinstance(state, dict) else state.user_request

    try:
        plan = await plan_workflow(
            user_request=user_req,
            llm=llm,
            registry=registry,
            max_steps=max_steps,
        )
        order = plan.get_execution_order()
        return {
            "execution_plan": plan,
            "workflow_status": WorkflowStatus.EXECUTING if order else WorkflowStatus.COMPLETED,
            "pending_steps": list(order),
            "completed_steps": [],
            "current_step_id": None,
        }
    except Exception as exc:
        logger.error("Planner node failed: %s", exc)
        curr_errors = (
            state.get("error_history", []) if isinstance(state, dict) else list(state.error_history)
        )
        return {
            "workflow_status": WorkflowStatus.FAILED,
            "error_history": curr_errors + [f"Planning error: {exc}"],
        }
