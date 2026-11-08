"""State management for SYNQRA agent workflows.

Provides strongly-typed, LangGraph-compatible state models preserving intermediate
execution results, pending approvals, execution plans, and error histories without
persisting secrets or sensitive credentials.
"""

import uuid
from datetime import UTC, datetime
from typing import Any, TypedDict

from pydantic import BaseModel, Field, model_validator

from synqra.agents.exceptions import InvalidAgentStateError
from synqra.agents.schemas import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStatus,
    ExecutionPlan,
    PlanStep,
    StepStatus,
    ToolExecutionResult,
    WorkflowStatus,
)


def utc_now() -> datetime:
    """Return the current timezone-aware UTC datetime."""
    return datetime.now(UTC)


class AgentStateDict(TypedDict, total=False):
    """LangGraph-compatible TypedDict representation of agent state."""

    workflow_id: str
    thread_id: str
    user_request: str
    execution_plan: dict[str, Any] | None
    current_step_id: str | None
    completed_steps: list[str]
    pending_steps: list[str]
    execution_results: dict[str, dict[str, Any]]
    approval_requests: list[dict[str, Any]]
    workflow_status: str
    retry_count: int
    error_history: list[str]
    final_response: str | None
    created_at: str
    updated_at: str


class AgentState(BaseModel):
    """Maintains the full execution context and progress for a SYNQRA agent workflow."""

    workflow_id: str = Field(
        default_factory=lambda: f"wf_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this workflow run.",
    )
    thread_id: str = Field(
        default_factory=lambda: f"th_{uuid.uuid4().hex[:12]}",
        description="Thread identifier for conversation or checkpoint grouping.",
    )
    user_request: str = Field(description="Original user request being processed.")
    execution_plan: ExecutionPlan | None = Field(
        default=None,
        description="Generated multi-step execution plan if planning has occurred.",
    )
    current_step_id: str | None = Field(
        default=None,
        description="ID of the step currently undergoing execution or approval.",
    )
    completed_steps: list[str] = Field(
        default_factory=list,
        description="List of step_ids successfully finished.",
    )
    pending_steps: list[str] = Field(
        default_factory=list,
        description="List of step_ids remaining to execute.",
    )
    execution_results: dict[str, ToolExecutionResult] = Field(
        default_factory=dict,
        description="Map of step_id to ToolExecutionResult records.",
    )
    approval_requests: list[ApprovalRequest] = Field(
        default_factory=list,
        description="History and active list of human approval requests.",
    )
    workflow_status: WorkflowStatus = Field(
        default=WorkflowStatus.PENDING,
        description="Current lifecycle status of the workflow.",
    )
    retry_count: int = Field(
        default=0,
        ge=0,
        description="Number of retries attempted across steps in this workflow.",
    )
    error_history: list[str] = Field(
        default_factory=list,
        description="Historical log of errors encountered during workflow execution.",
    )
    final_response: str | None = Field(
        default=None,
        description="Synthesized final response text to return to the user.",
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        description="Timestamp when the workflow was initialized.",
    )
    updated_at: datetime = Field(
        default_factory=utc_now,
        description="Timestamp of the most recent state mutation.",
    )

    @model_validator(mode="after")
    def validate_state_consistency(self) -> "AgentState":
        """Verify that completed and pending step sets are mutually consistent."""
        completed_set = set(self.completed_steps)
        overlap = completed_set.intersection(self.pending_steps)
        if overlap:
            raise InvalidAgentStateError(
                f"State inconsistency: steps cannot be both completed and pending: "
                f"{sorted(overlap)}."
            )
        return self

    def set_execution_plan(self, plan: ExecutionPlan) -> None:
        """Associate a newly generated plan and populate pending steps in topological order."""
        self.execution_plan = plan
        order = plan.get_execution_order()
        self.pending_steps = list(order)
        self.completed_steps = []
        self.current_step_id = None
        self.workflow_status = WorkflowStatus.EXECUTING if order else WorkflowStatus.COMPLETED
        self.updated_at = utc_now()

    def get_step(self, step_id: str) -> PlanStep | None:
        """Retrieve a plan step by ID from the associated execution plan."""
        if not self.execution_plan:
            return None
        return self.execution_plan.get_step(step_id)

    def get_next_runnable_steps(self) -> list[PlanStep]:
        """Return all pending steps whose dependencies have all completed successfully."""
        if not self.execution_plan or not self.pending_steps:
            return []

        completed_set = set(self.completed_steps)
        runnable: list[PlanStep] = []

        for step_id in self.pending_steps:
            step = self.execution_plan.get_step(step_id)
            if step and set(step.dependencies).issubset(completed_set):
                runnable.append(step)

        return runnable

    def mark_step_running(self, step_id: str) -> None:
        """Mark a step as actively running."""
        step = self.get_step(step_id)
        if not step:
            raise InvalidAgentStateError(f"Step '{step_id}' not found in plan.")
        step.transition_status(StepStatus.RUNNING)
        self.current_step_id = step_id
        self.workflow_status = WorkflowStatus.EXECUTING
        self.updated_at = utc_now()

    def mark_step_completed(self, step_id: str, result: ToolExecutionResult) -> None:
        """Record successful step completion, moving step from pending to completed."""
        step = self.get_step(step_id)
        if not step:
            raise InvalidAgentStateError(f"Step '{step_id}' not found in plan.")

        step.transition_status(StepStatus.COMPLETED)
        step.execution_result = result
        self.execution_results[step_id] = result

        if step_id in self.pending_steps:
            self.pending_steps.remove(step_id)
        if step_id not in self.completed_steps:
            self.completed_steps.append(step_id)

        if self.current_step_id == step_id:
            self.current_step_id = None

        if not self.pending_steps:
            self.workflow_status = WorkflowStatus.COMPLETED

        self.updated_at = utc_now()

    def mark_step_failed(self, step_id: str, error_message: str) -> None:
        """Record a failure for a step, logging error history and transitioning workflow state."""
        step = self.get_step(step_id)
        if not step:
            raise InvalidAgentStateError(f"Step '{step_id}' not found in plan.")

        step.transition_status(StepStatus.FAILED)
        step.error = error_message
        self.error_history.append(f"Step '{step_id}' failed: {error_message}")
        self.workflow_status = WorkflowStatus.FAILED
        self.updated_at = utc_now()

    def request_approval(self, request: ApprovalRequest) -> None:
        """Transition workflow into AWAITING_APPROVAL and track the pending approval request."""
        step = self.get_step(request.workflow_id) or self.get_step(self.current_step_id or "")
        if step and step.status == StepStatus.PENDING:
            step.transition_status(StepStatus.AWAITING_APPROVAL)

        self.approval_requests.append(request)
        self.workflow_status = WorkflowStatus.AWAITING_APPROVAL
        self.updated_at = utc_now()

    def apply_approval_decision(self, decision: ApprovalDecision) -> None:
        """Apply a human operator's decision to the matching approval request."""
        target: ApprovalRequest | None = None
        for req in self.approval_requests:
            if req.approval_id == decision.approval_id:
                target = req
                break

        if not target:
            raise InvalidAgentStateError(
                f"No matching approval request found for approval_id '{decision.approval_id}'."
            )

        if target.status != ApprovalStatus.PENDING:
            raise InvalidAgentStateError(
                f"Approval request '{decision.approval_id}' is already {target.status.value}."
            )

        if decision.approved:
            target.status = ApprovalStatus.APPROVED
            self.workflow_status = WorkflowStatus.EXECUTING
        else:
            target.status = ApprovalStatus.REJECTED
            self.workflow_status = WorkflowStatus.CANCELLED
            reason_text = decision.reason or "No reason provided."
            self.error_history.append(f"Approval rejected by {decision.decided_by}: {reason_text}")

        self.updated_at = utc_now()

    def is_resumable(self) -> bool:
        """Determine whether the workflow is in a state that can safely resume execution."""
        return self.workflow_status in (
            WorkflowStatus.AWAITING_APPROVAL,
            WorkflowStatus.EXECUTING,
            WorkflowStatus.PENDING,
        ) and bool(self.pending_steps)

    def to_langgraph_dict(self) -> AgentStateDict:
        """Convert state to a LangGraph-compatible dictionary."""
        return AgentStateDict(
            workflow_id=self.workflow_id,
            thread_id=self.thread_id,
            user_request=self.user_request,
            execution_plan=self.execution_plan.model_dump(mode="json")
            if self.execution_plan
            else None,
            current_step_id=self.current_step_id,
            completed_steps=list(self.completed_steps),
            pending_steps=list(self.pending_steps),
            execution_results={
                k: v.model_dump(mode="json") for k, v in self.execution_results.items()
            },
            approval_requests=[r.model_dump(mode="json") for r in self.approval_requests],
            workflow_status=self.workflow_status.value,
            retry_count=self.retry_count,
            error_history=list(self.error_history),
            final_response=self.final_response,
            created_at=self.created_at.isoformat(),
            updated_at=self.updated_at.isoformat(),
        )
