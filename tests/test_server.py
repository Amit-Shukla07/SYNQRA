"""Unit and integration tests for the SYNQRA MCP server."""

import json

import pytest
from starlette.testclient import TestClient

from synqra.core.config import Settings
from synqra.server.app import parse_args
from synqra.server.mcp_server import HealthCheckResponse, create_mcp_server


def test_configuration_defaults() -> None:
    """Verify default configuration settings for SYNQRA."""
    settings = Settings()

    assert settings.app_name == "SYNQRA"
    assert settings.app_version == "0.1.0"
    assert settings.host == "127.0.0.1"
    assert settings.port == 8000
    assert settings.debug is False
    assert settings.environment == "development"
    assert settings.transport == "streamable-http"
    assert settings.mcp_path == "/mcp"


def test_configuration_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify environment variable overrides work with aliases."""
    monkeypatch.setenv("SYNQRA_PORT", "9090")
    monkeypatch.setenv("SYNQRA_HOST", "0.0.0.0")
    monkeypatch.setenv("SYNQRA_DEBUG", "true")
    monkeypatch.setenv("SYNQRA_ENVIRONMENT", "production")

    settings = Settings()
    assert settings.port == 9090
    assert settings.host == "0.0.0.0"
    assert settings.debug is True
    assert settings.environment == "production"


def test_mcp_server_initialization() -> None:
    """Verify FastMCP server instance initialization and metadata."""
    settings = Settings(app_name="SYNQRA-Test", app_version="0.2.0")
    server = create_mcp_server(settings)

    assert server.name == "SYNQRA-Test"
    assert server.version == "0.2.0"
    assert server.instructions is not None
    assert "SYNQRA" in server.instructions


@pytest.mark.asyncio
async def test_health_check_tool_registration() -> None:
    """Verify that the health_check tool is registered with the MCP server."""
    server = create_mcp_server()
    tools = await server.list_tools()

    tool_names = [tool.name for tool in tools]
    assert "health_check" in tool_names

    health_tool = next(t for t in tools if t.name == "health_check")
    assert health_tool.description is not None
    assert "diagnostic health check" in health_tool.description.lower()


@pytest.mark.asyncio
async def test_health_check_tool_execution() -> None:
    """Verify that calling the health_check tool returns valid structured content."""
    settings = Settings(app_name="SYNQRA", app_version="0.1.0", transport="streamable-http")
    server = create_mcp_server(settings)

    result = await server.call_tool("health_check", {})

    assert result.is_error is False
    assert result.structured_content is not None

    data = result.structured_content
    # Validate against Pydantic schema
    validated = HealthCheckResponse.model_validate(data)

    assert validated.server_name == "SYNQRA"
    assert validated.status == "healthy"
    assert validated.version == "0.1.0"
    assert validated.available_integrations == []
    assert "youtube" in validated.planned_integrations
    assert "google_drive" in validated.planned_integrations
    assert "github" in validated.planned_integrations
    assert "google_calendar" in validated.planned_integrations
    assert "web_search" in validated.planned_integrations
    assert "gmail" in validated.planned_integrations
    assert len(validated.integrations_status) == 6
    assert validated.integrations_status["youtube"].is_configured is False
    assert validated.transport == "streamable-http"
    assert validated.environment == "development"


def test_streamable_http_handshake_and_tool_call() -> None:
    """Integration test: Verify streamable HTTP transport initialize & tools/call flow."""
    server = create_mcp_server()
    app = server.http_app(transport="streamable-http")

    with TestClient(app) as client:
        # 1. Initialize MCP Session
        init_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0"},
            },
        }
        init_res = client.post(
            "/mcp",
            json=init_payload,
            headers={"Accept": "application/json, text/event-stream"},
        )
        assert init_res.status_code == 200
        assert "serverInfo" in init_res.text
        assert "SYNQRA" in init_res.text

        # Extract session header if present
        session_id = init_res.headers.get("mcp-session-id")
        headers = {"Accept": "application/json, text/event-stream"}
        if session_id:
            headers["mcp-session-id"] = session_id

        # 2. Call health_check tool via HTTP JSON-RPC
        tool_call_payload = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "health_check", "arguments": {}},
        }
        call_res = client.post("/mcp", json=tool_call_payload, headers=headers)
        assert call_res.status_code == 200

        # FastMCP streamable-http returns Server-Sent Events (SSE) stream format
        lines = call_res.text.strip().split("\n")
        data_line = next(line for line in lines if line.startswith("data: "))
        json_data = json.loads(data_line[6:])

        assert json_data["id"] == 2
        structured = json_data["result"]["structuredContent"]
        assert structured["server_name"] == "SYNQRA"
        assert structured["status"] == "healthy"
        assert structured["version"] == "0.1.0"
        assert structured["available_integrations"] == []
        assert len(structured["planned_integrations"]) == 6
        assert len(structured["integrations_status"]) == 6


def test_stateless_http_request_without_session_id() -> None:
    """Verify that requests without an mcp-session-id succeed when stateless_http=True."""
    settings = Settings(stateless_http=True)
    server = create_mcp_server(settings)
    app = server.http_app(transport="streamable-http", stateless_http=True)

    with TestClient(app) as client:
        # Direct tools/list call without prior initialize or session header
        payload = {
            "jsonrpc": "2.0",
            "id": 42,
            "method": "tools/list",
            "params": {},
        }
        resp = client.post(
            "/mcp",
            json=payload,
            headers={"Accept": "application/json, text/event-stream"},
        )
        assert resp.status_code == 200
        assert "health_check" in resp.text

        # Verify JSON-RPC payload does not contain an error object
        lines = resp.text.strip().split("\n")
        data_line = next(line for line in lines if line.startswith("data: "))
        json_data = json.loads(data_line[6:])
        assert "error" not in json_data
        assert "result" in json_data


def test_browser_get_endpoints_return_helpful_info() -> None:
    """Verify that browser GET requests to / and /mcp return 200 OK with guidance."""
    server = create_mcp_server()
    app = server.http_app(transport="streamable-http", stateless_http=True)

    with TestClient(app) as client:
        # GET /
        root_res = client.get("/")
        assert root_res.status_code == 200
        root_json = root_res.json()
        assert root_json["server"] == "SYNQRA"
        assert root_json["status"] == "operational"
        assert root_json["mcp_endpoint"] == "/mcp"

        # GET /mcp (what a browser opens)
        mcp_res = client.get("/mcp")
        assert mcp_res.status_code == 200
        mcp_json = mcp_res.json()
        assert mcp_json["status"] == "operational"
        assert "active" in mcp_json["message"].lower()


def test_cli_argument_parsing() -> None:
    """Verify CLI argument parser functionality."""
    args = parse_args(["--host", "0.0.0.0", "--port", "9000", "--debug"])
    assert args.host == "0.0.0.0"
    assert args.port == 9000
    assert args.debug is True

    default_args = parse_args([])
    assert default_args.host is None
    assert default_args.port is None
    assert default_args.debug is None
