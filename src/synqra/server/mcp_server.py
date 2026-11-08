"""FastMCP server definition and core tools for SYNQRA.

Defines the FastMCP application instance, registers core diagnostic tools,
and manages server metadata.
"""

from typing import Any, Literal

from fastmcp import FastMCP
from pydantic import BaseModel, Field
from starlette.requests import Request
from starlette.responses import JSONResponse

from synqra.agents.approval.api import (
    handle_approve_request,
    handle_get_approval,
    handle_list_pending_approvals,
    handle_reject_request,
)
from synqra.core.config import Settings, get_settings
from synqra.integrations import (
    IntegrationRegistry,
    IntegrationStatus,
    default_registry,
)


class HealthCheckResponse(BaseModel):
    """Structured response model for the health_check MCP tool."""

    server_name: str = Field(description="Name of the MCP server.")
    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        default="healthy", description="Operational status of the server."
    )
    version: str = Field(description="Release version of the server.")
    available_integrations: list[str] = Field(
        default_factory=list,
        description="List of active and operational integration names. Empty until implemented.",
    )
    planned_integrations: list[str] = Field(
        default_factory=list,
        description="Integrations planned for upcoming phases.",
    )
    integrations_status: dict[str, IntegrationStatus] = Field(
        default_factory=dict,
        description="Detailed diagnostic health status for each registered integration.",
    )
    transport: str = Field(description="Configured transport mechanism for the server.")
    environment: str = Field(description="Current deployment environment.")


def create_mcp_server(
    settings: Settings | None = None,
    registry: IntegrationRegistry | None = None,
) -> FastMCP:
    """Create and configure a new FastMCP server instance for SYNQRA.

    Args:
        settings: Application settings. If None, default settings will be loaded.
        registry: Integration registry. If None, default_registry will be used.

    Returns:
        Configured FastMCP instance with registered tools.
    """
    cfg = settings or get_settings()
    reg = registry if registry is not None else default_registry

    server = FastMCP(
        name=cfg.app_name,
        version=cfg.app_version,
        instructions=(
            "SYNQRA is a universal integration MCP server connecting AI agents "
            "with external productivity and content services."
        ),
    )

    # Mount all configured integrations (e.g. Web Search)
    reg.mount_all(server)

    @server.tool(
        name="health_check",
        description=(
            "Perform a diagnostic health check of the SYNQRA MCP server, returning "
            "server status, version, configured transport, and integration readiness."
        ),
    )
    async def health_check() -> HealthCheckResponse:
        """Return operational health status and integration availability."""
        statuses = await reg.check_health_all()
        return HealthCheckResponse(
            server_name=cfg.app_name,
            status="healthy",
            version=cfg.app_version,
            available_integrations=reg.list_available(),
            planned_integrations=reg.list_planned(),
            integrations_status=statuses,
            transport=cfg.transport,
            environment=cfg.environment,
        )

    @server.tool(
        name="synqra_run_agent",
        description=(
            "Execute a multi-step natural language agent workflow using SYNQRA's "
            "LangGraph orchestration engine across connected tools (GitHub, Gmail, "
            "Google Calendar, Google Drive, YouTube, Web Search). Protected WRITE or "
            "SENSITIVE operations pause safely for Human-in-the-Loop authorization."
        ),
    )
    async def synqra_run_agent(
        request: str,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute an agent workflow from a natural language request.

        Args:
            request: The natural language task or instructions for the agent.
            thread_id: Optional thread ID to maintain state across multi-turn interactions.
        """
        from synqra.cli.commands import run_agent_request

        return await run_agent_request(
            request=request,
            thread_id=thread_id,
            silent=True,
        )

    @server.custom_route("/", methods=["GET"])
    async def root_info(_request: Request) -> JSONResponse:
        """Informational endpoint for browsers and health probes."""
        return JSONResponse(
            {
                "server": cfg.app_name,
                "version": cfg.app_version,
                "status": "operational",
                "mcp_endpoint": cfg.mcp_path,
                "transport": cfg.transport,
                "stateless_http": cfg.stateless_http,
                "instructions": (
                    f"SYNQRA MCP server is operational. Connect with an MCP client "
                    f"(Cursor, Antigravity, Claude) via POST to {cfg.mcp_path}."
                ),
            }
        )

    @server.custom_route(cfg.mcp_path, methods=["GET"])
    async def mcp_get_info(_request: Request) -> JSONResponse:
        """Helpful response when a web browser sends a GET request to the MCP endpoint."""
        return JSONResponse(
            {
                "server": cfg.app_name,
                "status": "operational",
                "endpoint": cfg.mcp_path,
                "message": (
                    f"SYNQRA {cfg.mcp_path} Streamable HTTP endpoint is active. "
                    "Web browsers issue GET requests, whereas MCP protocol uses POST for "
                    "JSON-RPC 2.0. Configure this URL directly in your MCP client."
                ),
            }
        )

    # ---------------------------------------------------------
    # Human-in-the-Loop Approval Endpoints (Phase 9.4)
    # ---------------------------------------------------------
    @server.custom_route("/api/approvals", methods=["GET"])
    async def api_list_approvals(request: Request) -> JSONResponse:
        """List active pending approval requests."""
        return await handle_list_pending_approvals(request)

    @server.custom_route("/api/approvals/{approval_id}", methods=["GET"])
    async def api_get_approval(request: Request) -> JSONResponse:
        """Get full details and operation arguments for an approval request."""
        return await handle_get_approval(request)

    @server.custom_route("/api/approvals/{approval_id}/approve", methods=["POST"])
    async def api_approve_request(request: Request) -> JSONResponse:
        """Authorize an operation by an authenticated human operator."""
        return await handle_approve_request(request)

    @server.custom_route("/api/approvals/{approval_id}/reject", methods=["POST"])
    async def api_reject_request(request: Request) -> JSONResponse:
        """Reject an operation, preventing tool execution."""
        return await handle_reject_request(request)

    return server


# Default module-level server instance
mcp_server: FastMCP = create_mcp_server()
