"""Result Aggregator Node for SYNQRA LangGraph Agentic Engine.

Collects and consolidates outputs from completed steps across integrations,
preserves URLs, references, and document links, accurately reports partial failures,
and produces a clear, structured final response for the user.
"""

import contextlib
import json
import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from synqra.agents.schemas import ExecutionPlan, StepStatus, WorkflowStatus
from synqra.agents.state import AgentState

logger = logging.getLogger("synqra.agents.aggregator")

AGGREGATOR_SYSTEM_PROMPT = """You are the SYNQRA Result Aggregator Agent.
Your task is to take the user's original request and the executed tool outputs,
and synthesize a clear, comprehensive, and well-structured answer.

Strict guidelines:
1. Synthesize the findings into an insightful, direct response.
2. DO NOT invent facts or hallucinate details not present in the tool results.
3. Explicitly preserve and include all relevant URLs, repository links, and citations.
4. If any step failed or paused for approval, accurately state that status.
"""


def extract_links_and_citations(results: dict[str, Any]) -> list[str]:
    """Scan structured results recursively for URLs, IDs, and citations."""
    links: list[str] = []

    def _walk(obj: Any) -> None:
        if hasattr(obj, "model_dump"):
            with contextlib.suppress(Exception):
                obj = obj.model_dump()
        elif hasattr(obj, "__dict__"):
            with contextlib.suppress(Exception):
                obj = vars(obj)

        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in ("url", "web_view_link", "html_url", "watch_url", "link") and isinstance(
                    v, str
                ):
                    if v not in links:
                        links.append(v)
                elif (
                    isinstance(v, (dict, list))
                    or hasattr(v, "model_dump")
                    or hasattr(v, "__dict__")
                ):
                    _walk(v)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(results)
    return links


def format_deterministic_response(
    user_request: str,
    plan: ExecutionPlan | None,
    results: dict[str, Any],
    error_history: list[str],
    status: WorkflowStatus | str,
) -> str:
    """Construct an honest, deterministic text summary of the workflow execution."""
    lines: list[str] = [
        "### SYNQRA Workflow Execution Summary",
        f"**Request:** {user_request}\n",
    ]

    status_str = status.value if hasattr(status, "value") else str(status)
    lines.append(f"**Status:** `{status_str.upper()}`\n")

    if plan:
        lines.append("#### Completed Steps:")
        for step in plan.steps:
            icon = (
                "✅"
                if step.status == StepStatus.COMPLETED
                else ("⏳" if step.status == StepStatus.PENDING else "❌")
            )
            lines.append(f"- {icon} **{step.step_id}** ({step.tool_name}): {step.description}")
            if step.status == StepStatus.FAILED and step.error:
                lines.append(f"  *Error:* {step.error}")

    # Extract all discovered citations and URLs
    citations = extract_links_and_citations(results)
    if citations:
        lines.append("\n#### Discovered Sources & Links:")
        for cite in citations:
            lines.append(f"- {cite}")

    # Report errors if any
    if error_history:
        lines.append("\n#### Diagnostics & Errors:")
        for err in error_history:
            lines.append(f"- {err}")

    return "\n".join(lines)


async def aggregate_results(
    user_request: str,
    plan: ExecutionPlan | None,
    results: dict[str, Any],
    error_history: list[str],
    status: WorkflowStatus | str,
    llm: BaseChatModel | None = None,
) -> str:
    """Synthesize results into a cohesive final response, with optional LLM enhancement."""
    deterministic_summary = format_deterministic_response(
        user_request=user_request,
        plan=plan,
        results=results,
        error_history=error_history,
        status=status,
    )

    if llm is None:
        return deterministic_summary

    # Use LLM to enhance readability while grounding in deterministic data
    prompt_payload = {
        "user_request": user_request,
        "status": status.value if hasattr(status, "value") else str(status),
        "execution_results": results,
        "error_history": error_history,
    }

    try:
        messages = [
            SystemMessage(content=AGGREGATOR_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"User Request: {user_request}\n\n"
                    f"Execution Context:\n{json.dumps(prompt_payload, indent=2, default=str)}\n\n"
                    f"Synthesize the final response following the strict guidelines."
                )
            ),
        ]
        resp = await llm.ainvoke(messages)
        content = resp.content if hasattr(resp, "content") else str(resp)
        synthesized = str(content).strip() if content else deterministic_summary

        # Guarantee preservation of extracted links/citations and diagnostics
        links = extract_links_and_citations(results)
        missing_links = [lnk for lnk in links if lnk not in synthesized]
        sections = [synthesized]

        if missing_links:
            ref_lines = ["\n### References & Links"]
            for lnk in missing_links:
                ref_lines.append(f"- {lnk}")
            sections.append("\n".join(ref_lines))

        if "### Diagnostics" not in synthesized:
            status_str = status.value if hasattr(status, "value") else str(status)
            diag_lines = [
                "\n### Diagnostics",
                f"- Status: {status_str}",
            ]
            if plan and plan.steps:
                completed = sum(1 for s in plan.steps if s.status == StepStatus.COMPLETED)
                diag_lines.append(f"- Steps executed: {completed}/{len(plan.steps)}")
            if error_history:
                diag_lines.append(f"- Errors encountered: {len(error_history)}")
            sections.append("\n".join(diag_lines))

        return "\n\n".join(sections).strip()
    except Exception as exc:
        logger.warning("LLM aggregation failed, falling back to deterministic summary: %s", exc)
        return deterministic_summary


async def aggregator_node(
    state: AgentState | dict[str, Any],
    config: RunnableConfig | None = None,
    llm: BaseChatModel | None = None,
) -> dict[str, Any]:
    """LangGraph node consolidating final outputs and setting final_response."""
    _ = config
    if isinstance(state, dict):
        user_req = state.get("user_request", "")
        plan_raw = state.get("execution_plan")
        plan = ExecutionPlan.model_validate(plan_raw) if isinstance(plan_raw, dict) else plan_raw
        results = dict(state.get("execution_results", {}))
        errors = list(state.get("error_history", []))
        status = state.get("workflow_status", WorkflowStatus.COMPLETED)
    else:
        user_req = state.user_request
        plan = state.execution_plan
        results = dict(state.execution_results)
        errors = list(state.error_history)
        status = state.workflow_status

    final_text = await aggregate_results(
        user_request=user_req,
        plan=plan,
        results=results,
        error_history=errors,
        status=status,
        llm=llm,
    )

    return {
        "final_response": final_text,
        "workflow_status": status,
    }
