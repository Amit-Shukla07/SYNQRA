"""SYNQRA Agentic Orchestration Package.

Provides core schemas, state management, LLM factory, tool registration, adapter,
permission checking, and LangGraph workflow orchestration for autonomous multi-step
problem solving across SYNQRA's service integrations.
"""

from synqra.agents.aggregator import (
    aggregate_results,
    aggregator_node,
    format_deterministic_response,
)
from synqra.agents.checkpointer import (
    PostgresCheckpointerManager,
    close_checkpointer,
    get_checkpointer,
    get_postgres_checkpointer_manager,
    mask_database_url,
    normalize_postgres_url,
)
from synqra.agents.evaluator import (
    EvaluationResult,
    EvaluatorAction,
    evaluate_workflow_state,
    evaluator_node,
)
from synqra.agents.exceptions import (
    ApprovalError,
    CheckpointerConfigurationError,
    CheckpointerConnectionError,
    CheckpointerError,
    InvalidAgentStateError,
    InvalidExecutionPlanError,
    LLMConfigurationError,
    LLMProviderError,
    SynqraAgentError,
    WorkflowRecoveryError,
)
from synqra.agents.executor import execute_plan_step, executor_node
from synqra.agents.graph import create_agent_graph, resume_agent_workflow, run_agent_workflow
from synqra.agents.llm import MockChatModel, get_llm, get_mock_llm
from synqra.agents.permissions import (
    TOOL_PERMISSIONS,
    TOOLS_REQUIRING_CONFIRMATION,
    ApprovalContext,
    get_tool_permission_level,
    validate_approval_for_execution,
)
from synqra.agents.planner import (
    PlannerOutput,
    PlanStepProposal,
    plan_workflow,
    planner_node,
)
from synqra.agents.recovery import recovery_node
from synqra.agents.router import ScheduledBatch, route_runnable_steps, router_node
from synqra.agents.schemas import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStatus,
    ExecutionPlan,
    LLMProvider,
    PermissionLevel,
    PlanStep,
    StepStatus,
    ToolExecutionResult,
    WorkflowStatus,
)
from synqra.agents.state import AgentState, AgentStateDict
from synqra.agents.tool_adapter import AgentToolAdapter
from synqra.agents.tool_registry import AgentToolRegistry, RegisteredTool
from synqra.agents.tools import (
    build_agent_tool_registry,
    execute_tool,
    get_default_agent_tool_registry,
    get_default_tool_adapter,
    get_tool,
    get_tool_schema,
    get_tools_by_integration,
    get_tools_by_permission,
    list_tools,
)
from synqra.agents.workflow_recovery import (
    RecoveredThreadStatus,
    WorkflowRecoveryService,
)

__all__ = [
    "AgentState",
    "AgentStateDict",
    "AgentToolAdapter",
    "AgentToolRegistry",
    "ApprovalContext",
    "ApprovalDecision",
    "ApprovalError",
    "ApprovalRequest",
    "ApprovalStatus",
    "CheckpointerConfigurationError",
    "CheckpointerConnectionError",
    "CheckpointerError",
    "EvaluationResult",
    "EvaluatorAction",
    "ExecutionPlan",
    "InvalidAgentStateError",
    "InvalidExecutionPlanError",
    "LLMConfigurationError",
    "LLMProvider",
    "LLMProviderError",
    "MockChatModel",
    "PermissionLevel",
    "PlanStep",
    "PlanStepProposal",
    "PlannerOutput",
    "PostgresCheckpointerManager",
    "RecoveredThreadStatus",
    "RegisteredTool",
    "ScheduledBatch",
    "StepStatus",
    "SynqraAgentError",
    "TOOL_PERMISSIONS",
    "TOOLS_REQUIRING_CONFIRMATION",
    "ToolExecutionResult",
    "WorkflowRecoveryError",
    "WorkflowRecoveryService",
    "WorkflowStatus",
    "aggregate_results",
    "aggregator_node",
    "build_agent_tool_registry",
    "close_checkpointer",
    "create_agent_graph",
    "evaluate_workflow_state",
    "evaluator_node",
    "execute_plan_step",
    "execute_tool",
    "executor_node",
    "format_deterministic_response",
    "get_checkpointer",
    "get_default_agent_tool_registry",
    "get_default_tool_adapter",
    "get_llm",
    "get_mock_llm",
    "get_postgres_checkpointer_manager",
    "get_tool",
    "get_tool_permission_level",
    "get_tool_schema",
    "get_tools_by_integration",
    "get_tools_by_permission",
    "list_tools",
    "mask_database_url",
    "normalize_postgres_url",
    "plan_workflow",
    "planner_node",
    "recovery_node",
    "route_runnable_steps",
    "router_node",
    "resume_agent_workflow",
    "run_agent_workflow",
    "validate_approval_for_execution",
]
