"""Pydantic v2 schemas and data models for SYNQRA agentic orchestration.

Defines typed, immutable schemas for multi-step execution plans, individual plan steps,
tool execution results, approval requests, approval decisions, and workflow statuses.
"""

import uuid
from collections import defaultdict, deque
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from synqra.agents.exceptions import ApprovalError, InvalidExecutionPlanError


def utc_now() -> datetime:
    """Return the current timezone-aware UTC datetime."""
    return datetime.now(UTC)


class WorkflowStatus(StrEnum):
    """Lifecycle status of a SYNQRA agent workflow."""

    PENDING = "pending"
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class StepStatus(StrEnum):
    """Execution status of an individual plan step."""

    PENDING = "pending"
    AWAITING_APPROVAL = "awaiting_approval"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    REJECTED = "rejected"


class PermissionLevel(StrEnum):
    """Required authorization/safety classification for tool operations."""

    READ = "read"
    WRITE = "write"
    SENSITIVE = "sensitive"
    ADMIN = "admin"


class ApprovalStatus(StrEnum):
    """Status of a human-in-the-loop approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CONSUMED = "consumed"


class LLMProvider(StrEnum):
    """Supported LLM providers for planning and reasoning."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class ToolExecutionResult(BaseModel):
    """Records the structured outcome of executing an individual tool."""

    model_config = ConfigDict(frozen=True)

    execution_id: str = Field(
        default_factory=lambda: f"exec_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this tool execution.",
    )
    tool_name: str = Field(description="Name of the SYNQRA tool executed.")
    success: bool = Field(description="Whether the tool execution succeeded.")
    output: Any | None = Field(default=None, description="Structured tool output data.")
    error: str | None = Field(default=None, description="Error message if execution failed.")
    execution_duration: float | None = Field(
        default=None,
        description="Duration of tool execution in seconds.",
    )
    started_at: datetime = Field(
        default_factory=utc_now,
        description="Timestamp when execution began.",
    )
    completed_at: datetime = Field(
        default_factory=utc_now,
        description="Timestamp when execution concluded.",
    )


VALID_STEP_TRANSITIONS: dict[StepStatus, set[StepStatus]] = {
    StepStatus.PENDING: {
        StepStatus.RUNNING,
        StepStatus.AWAITING_APPROVAL,
        StepStatus.SKIPPED,
        StepStatus.FAILED,
        StepStatus.REJECTED,
    },
    StepStatus.AWAITING_APPROVAL: {
        StepStatus.RUNNING,
        StepStatus.SKIPPED,
        StepStatus.FAILED,
        StepStatus.PENDING,
        StepStatus.REJECTED,
    },
    StepStatus.RUNNING: {
        StepStatus.COMPLETED,
        StepStatus.FAILED,
        StepStatus.AWAITING_APPROVAL,
    },
    StepStatus.COMPLETED: set(),
    StepStatus.FAILED: {
        StepStatus.PENDING,
        StepStatus.RUNNING,  # Allow retry
    },
    StepStatus.SKIPPED: set(),
    StepStatus.REJECTED: set(),
}


class PlanStep(BaseModel):
    """Represents a discrete task within an execution plan."""

    step_id: str = Field(description="Unique identifier for the step (e.g. 'step_1').")
    description: str = Field(description="Human-readable explanation of this step's purpose.")
    tool_name: str = Field(description="Name of the SYNQRA tool to be invoked.")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value arguments to pass into the tool.",
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="List of step_ids that must complete successfully before this step executes.",
    )
    status: StepStatus = Field(
        default=StepStatus.PENDING,
        description="Current execution status of the step.",
    )
    permission_level: PermissionLevel = Field(
        default=PermissionLevel.READ,
        description="Permission level required to execute this step.",
    )
    execution_result: ToolExecutionResult | None = Field(
        default=None,
        description="Result record once execution is attempted.",
    )
    error: str | None = Field(
        default=None,
        description="Error description if step execution failed.",
    )

    @model_validator(mode="after")
    def validate_self_dependency(self) -> "PlanStep":
        """Prevent a step from depending on itself."""
        if self.step_id in self.dependencies:
            raise InvalidExecutionPlanError(
                f"Step '{self.step_id}' cannot depend on itself (self-dependency)."
            )
        return self

    def transition_status(self, new_status: StepStatus) -> None:
        """Validate and apply a status transition for this step."""
        allowed = VALID_STEP_TRANSITIONS.get(self.status, set())
        if new_status not in allowed:
            raise InvalidExecutionPlanError(
                f"Invalid status transition for step '{self.step_id}': "
                f"cannot transition from {self.status.value} to {new_status.value}."
            )
        self.status = new_status


class ExecutionPlan(BaseModel):
    """Represents the complete multi-step execution plan generated by the planner agent."""

    plan_id: str = Field(
        default_factory=lambda: f"plan_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this plan.",
    )
    user_request: str = Field(description="The original user query or request being solved.")
    steps: list[PlanStep] = Field(
        default_factory=list,
        description="Ordered or interdependent tasks composing the plan.",
    )
    reasoning_summary: str | None = Field(
        default=None,
        description="Summary of the planner agent's reasoning and decomposition logic.",
    )
    estimated_steps: int = Field(
        default=0,
        description="Estimated total count of steps to execute.",
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        description="Timestamp when the plan was created.",
    )
    status: WorkflowStatus = Field(
        default=WorkflowStatus.PENDING,
        description="Overall workflow status of this plan.",
    )

    @model_validator(mode="after")
    def validate_plan_graph(self) -> "ExecutionPlan":
        """Validate step uniqueness, dependency validity, and absence of cycles."""
        if not self.steps:
            if self.estimated_steps == 0:
                self.estimated_steps = 0
            return self

        # 1. Step ID uniqueness
        seen_ids: set[str] = set()
        duplicate_ids: set[str] = set()
        for step in self.steps:
            if step.step_id in seen_ids:
                duplicate_ids.add(step.step_id)
            seen_ids.add(step.step_id)

        if duplicate_ids:
            raise InvalidExecutionPlanError(
                f"Duplicate step IDs found in execution plan: {sorted(duplicate_ids)}."
            )

        # 2. Dependency reference validity
        for step in self.steps:
            for dep in step.dependencies:
                if dep not in seen_ids:
                    raise InvalidExecutionPlanError(
                        f"Step '{step.step_id}' references non-existent dependency '{dep}'."
                    )

        # 3. Cycle detection using Kahn's algorithm (topological sort)
        in_degree: dict[str, int] = {s.step_id: 0 for s in self.steps}
        adj: dict[str, list[str]] = defaultdict(list)

        for step in self.steps:
            for dep in step.dependencies:
                adj[dep].append(step.step_id)
                in_degree[step.step_id] += 1

        queue = deque([sid for sid, deg in in_degree.items() if deg == 0])
        visited_count = 0

        while queue:
            curr = queue.popleft()
            visited_count += 1
            for neighbor in adj[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if visited_count != len(self.steps):
            cycle_nodes = [sid for sid, deg in in_degree.items() if deg > 0]
            raise InvalidExecutionPlanError(
                f"Circular dependency detected among steps: {sorted(cycle_nodes)}."
            )

        if self.estimated_steps == 0:
            self.estimated_steps = len(self.steps)

        return self

    def get_step(self, step_id: str) -> PlanStep | None:
        """Retrieve a step by its ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def get_execution_order(self) -> list[str]:
        """Return step IDs in a valid topological execution order."""
        in_degree: dict[str, int] = {s.step_id: len(s.dependencies) for s in self.steps}
        adj: dict[str, list[str]] = defaultdict(list)

        for step in self.steps:
            for dep in step.dependencies:
                adj[dep].append(step.step_id)

        queue = deque([sid for sid, deg in in_degree.items() if deg == 0])
        order: list[str] = []

        while queue:
            curr = queue.popleft()
            order.append(curr)
            for neighbor in adj[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return order

    def get_parallel_execution_tiers(self) -> list[list[str]]:
        """Return step IDs partitioned into parallel execution waves/tiers."""
        step_deps: dict[str, set[str]] = {s.step_id: set(s.dependencies) for s in self.steps}
        completed: set[str] = set()
        tiers: list[list[str]] = []

        remaining = set(step_deps.keys())
        while remaining:
            ready = [sid for sid in remaining if step_deps[sid].issubset(completed)]
            if not ready:
                break
            ready.sort()
            tiers.append(ready)
            completed.update(ready)
            remaining.difference_update(ready)

        return tiers


class ApprovalRequest(BaseModel):
    """Represents a request for human approval before executing a sensitive action."""

    approval_id: str = Field(
        default_factory=lambda: f"appr_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this approval request.",
    )
    workflow_id: str = Field(description="Identifier of the workflow requesting approval.")
    tool_name: str = Field(description="Exact tool targeted for execution.")
    arguments: dict[str, Any] = Field(description="Exact arguments to pass to the tool.")
    reason: str = Field(description="Justification explaining why approval is required.")
    permission_level: PermissionLevel = Field(
        default=PermissionLevel.SENSITIVE,
        description="Permission level classification triggering this request.",
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        description="Timestamp when the request was generated.",
    )
    expires_at: datetime | None = Field(
        default=None,
        description="Optional timestamp after which the request automatically expires.",
    )
    status: ApprovalStatus = Field(
        default=ApprovalStatus.PENDING,
        description="Current status of the approval request.",
    )

    @model_validator(mode="after")
    def validate_expiration(self) -> "ApprovalRequest":
        """Ensure expires_at is in the future relative to created_at."""
        if self.expires_at is not None and self.expires_at <= self.created_at:
            raise ApprovalError(
                f"expires_at ({self.expires_at}) must be after created_at ({self.created_at})."
            )
        return self

    def is_expired(self, at_time: datetime | None = None) -> bool:
        """Check whether the approval request has exceeded its expiry time."""
        if self.expires_at is None:
            return False
        check_time = at_time or utc_now()
        return check_time > self.expires_at


DISALLOWED_DECIDED_BY_IDENTIFIERS: frozenset[str] = frozenset(
    {
        "llm",
        "ai",
        "agent",
        "assistant",
        "system",
        "synqra_planner",
        "synqra_agent",
        "gpt-4o",
        "gpt-4o-mini",
        "claude",
        "gemini",
        "automated",
        "bot",
    }
)


class ApprovalDecision(BaseModel):
    """Represents a human operator's decision regarding a sensitive action."""

    approval_id: str = Field(description="The approval_id being decided upon.")
    approved: bool = Field(description="True if approved, False if rejected.")
    decided_at: datetime = Field(
        default_factory=utc_now,
        description="Timestamp when the decision was recorded.",
    )
    decided_by: str = Field(description="Identifier of the human operator rendering the decision.")
    reason: str | None = Field(
        default=None,
        description="Optional explanation or comment accompanying the decision.",
    )

    @model_validator(mode="after")
    def validate_human_decision_maker(self) -> "ApprovalDecision":
        """Ensure decisions are rendered by humans, never by automated LLM agents."""
        clean_decider = self.decided_by.strip().lower()
        if not clean_decider:
            raise ApprovalError("Approval decision must provide a non-empty decided_by identifier.")

        if clean_decider in DISALLOWED_DECIDED_BY_IDENTIFIERS:
            raise ApprovalError(
                f"Disallowed decider '{self.decided_by}': "
                "The LLM must never generate or approve its own approval decision."
            )
        return self
