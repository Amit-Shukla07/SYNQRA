# SYNQRA – The Universal MCP Integration Server

SYNQRA is a production-grade, modular Model Context Protocol (MCP) server engineered to connect AI agents and MCP-compliant clients (such as Google Antigravity, Claude Desktop, and Cursor) with external productivity, search, and storage services.

---

## 1. Technology Stack

* **Language**: Python 3.12+
* **MCP Framework**: [FastMCP 4.1.6](https://gofastmcp.com)
* **Agentic Orchestration (Planned)**: LangChain, LangGraph
* **Data Validation & Settings**: Pydantic v2 & Pydantic Settings
* **HTTP Client**: HTTPx
* **Database & ORM**: PostgreSQL, Psycopg (v3 binary), SQLAlchemy
* **Code Quality & Testing**: Pytest, Pytest-Asyncio, Ruff
* **External Integration Provider (Planned)**: Composio
* **Transport Protocol**: Streamable HTTP (SSE JSON-RPC 2.0 at `/mcp`) & Stdio

---

## 2. Architecture Overview

SYNQRA follows a clean, modular architecture separating configuration, server lifecycle, tools, and future external integrations:

```
synqra/
├── pyproject.toml              # Project dependencies, CLI entry points, and tool configs
├── .env.example                # Safe environment variable template (no credentials)
├── README.md                   # System documentation
├── src/
│   └── synqra/
│       ├── __init__.py         # Package root & high-level exports
│       ├── core/
│       │   ├── __init__.py
│       │   └── config.py       # Pydantic Settings configuration & env validation
│       ├── server/
│       │   ├── __init__.py
│       │   ├── mcp_server.py   # FastMCP server instance, schemas & tool definitions
│       │   └── app.py          # Application entry point, CLI parser & HTTP runner
│       ├── integrations/       # Service integration modules (Phase 2)
│       │   └── __init__.py
│       ├── agents/             # LangGraph agent workflows (Phase 2+)
│       │   └── __init__.py
│       └── database/           # Database models and session management (Phase 2+)
│           └── __init__.py
└── tests/
    └── test_server.py          # Pytest suite for config, tools, and HTTP transport
```

### Key Architectural Decisions:
1. **Server Factory Pattern (`create_mcp_server`)**: Decouples server instantiation from global state, allowing clean dependency injection of settings during unit and integration tests.
2. **Streamable HTTP Transport**: Implements the modern MCP Streamable HTTP transport standard (Server-Sent Events JSON-RPC 2.0) mounted at `/mcp`, allowing both local and networked AI agent access.
3. **Structured Tool Schemas**: Utilizes Pydantic v2 models for both input parameter validation and output responses, guaranteeing strict typing and contract adherence for LLM clients.

---

## 3. Installation & Setup

### Prerequisites
* Python >= 3.12
* [uv](https://docs.astral.sh/uv/) package manager (recommended) or standard `pip`

### Step 1: Clone and Navigate
```bash
cd /path/to/synqra
```

### Step 2: Install Dependencies & Setup Virtual Environment
Using `uv`:
```bash
uv sync
```
Or using standard Python venv:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## 4. Environment Configuration

Copy the example environment file and configure variables:
```bash
cp .env.example .env
```

Available environment variables:
| Variable | Default | Description |
| :--- | :--- | :--- |
| `SYNQRA_APP_NAME` | `SYNQRA` | Server identifier name |
| `SYNQRA_APP_VERSION` | `0.1.0` | Semantic version string |
| `SYNQRA_HOST` | `127.0.0.1` | Network interface to bind |
| `SYNQRA_PORT` | `8000` | Port for the HTTP server |
| `SYNQRA_DEBUG` | `false` | Enable verbose logging |
| `SYNQRA_ENVIRONMENT`| `development` | Deployment environment (`development`, `staging`, `production`, `testing`) |
| `SYNQRA_TRANSPORT` | `streamable-http`| FastMCP transport (`streamable-http`, `stdio`, `sse`, `http`) |
| `SYNQRA_MCP_PATH` | `/mcp` | Path mount for MCP Streamable HTTP |
| `SYNQRA_STATELESS_HTTP` | `true` | Enables stateless HTTP mode (no `mcp-session-id` required) |

---

## 5. Running the MCP Server

### Method A: Via CLI entry point
```bash
uv run synqra
```
Or with custom host and port flags:
```bash
uv run synqra --host 127.0.0.1 --port 8000 --debug
```

### Method B: Via Python Module
```bash
uv run python -m synqra.server.app
```

The Streamable HTTP MCP server will start at:
```
http://127.0.0.1:8000/mcp
```

---

## 6. Available MCP Tools

### `health_check`
Diagnostic probe that returns the server's health status, running version, configured transport, operational environment, and dynamic integration status.

* **Inputs**: None (`{}`)
* **Output Format**:
```json
{
  "server_name": "SYNQRA",
  "status": "healthy",
  "version": "0.1.0",
  "available_integrations": ["youtube"],
  "planned_integrations": [
    "google_drive",
    "github",
    "google_calendar",
    "web_search",
    "gmail"
  ],
  "integrations_status": {
    "youtube": {
      "name": "youtube",
      "display_name": "YouTube",
      "state": "connected",
      "is_configured": true,
      "is_connected": true,
      "registered_tools": ["search_youtube_videos", "get_video_details"]
    }
  },
  "transport": "streamable-http",
  "environment": "development"
}
```

### `search_youtube_videos`
Search YouTube for educational videos, lecture recordings, tutorials, and learning materials via Composio.

* **Inputs**:
  - `query` (string, required): Search query or keywords.
  - `max_results` (integer, optional, default: 5, max: 50): Number of videos to retrieve.
  - `order` (string, optional, default: "relevance"): Sort order (`relevance`, `date`, `viewCount`, `rating`).
* **Output Format**:
```json
{
  "query": "fastmcp tutorial",
  "total_results": 2,
  "videos": [
    {
      "video_id": "abc123xyz",
      "title": "Building MCP Servers with FastMCP in Python",
      "url": "https://www.youtube.com/watch?v=abc123xyz",
      "channel_name": "AI Engineers",
      "description": "Learn how to build production-grade Model Context Protocol servers.",
      "published_at": "2024-05-10T14:30:00Z",
      "duration": "18m 42s"
    }
  ]
}
```

### `get_video_details`
Retrieve detailed metadata for a YouTube video by its video ID.

* **Inputs**:
  - `video_id` (string, required): YouTube 11-character video ID.
* **Output Format**:
```json
{
  "video_id": "abc123xyz",
  "title": "Building MCP Servers with FastMCP in Python",
  "url": "https://www.youtube.com/watch?v=abc123xyz",
  "channel_name": "AI Engineers",
  "description": "Detailed video description and resources...",
  "published_at": "2024-05-10T14:30:00Z",
  "duration": "18m 42s",
  "view_count": 48200,
  "like_count": 3100
}
```

---

## 7. YouTube Integration & Composio Authentication

### Architecture
```
MCP Client (Cursor / Antigravity)
       │ (JSON-RPC 2.0 over Streamable HTTP)
       ▼
SYNQRA MCP Server (FastMCP)
       │ (SYNQRA Tools: search_youtube_videos, get_video_details)
       ▼
YouTubeIntegration Connector
       │ (Composio Session API: composio.create(user_id=..., toolkits=['youtube']))
       ▼
Composio Tool Router Session (session.authorize, session.execute)
       │ (OAuth 2.0 / User-Scoped Connection)
       ▼
YouTube Data API v3
```

### Configuration & Authentication Flow
1. Obtain an API key from [Composio](https://composio.dev).
2. Set the environment variable in `.env`:
   ```env
   SYNQRA_COMPOSIO_API_KEY="your-composio-api-key"
   SYNQRA_COMPOSIO_USER_ID="synqra_default_user"  # Stable user identifier
   ```
3. Connect your YouTube account:
   - When the server starts without an active connection, `health_check` reports `state="configured"` with an OAuth authorization link:
     ```
     YouTube account not connected. Authorize at: https://composio.dev/auth/connect/...
     ```
   - Click the link to complete the Google OAuth grant.
   - Alternatively, connect via the Composio CLI: `composio add youtube`
4. Start SYNQRA:
   ```bash
   uv run synqra
   ```
When credentials and an active connection are verified, `YouTubeIntegration` dynamically transitions to `state="connected"` and exposes `search_youtube_videos` and `get_video_details`.

---

## 7. Connecting SYNQRA to MCP Clients

### A. Google Antigravity
To connect SYNQRA as an MCP server in Google Antigravity, configure the MCP settings (in your Antigravity MCP configuration or workspace settings):

**Streamable HTTP Transport (Recommended):**
```json
{
  "mcpServers": {
    "synqra": {
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

**Stdio Transport (Direct Subprocess):**
```json
{
  "mcpServers": {
    "synqra": {
      "command": "/Users/amitshukla/Desktop/synqra/.venv/bin/python",
      "args": ["-m", "fastmcp", "run", "src/synqra/server/mcp_server.py:mcp_server", "--transport", "stdio"]
    }
  }
}
```

### B. Cursor IDE
Open Cursor Settings (`Cmd + ,` on macOS) -> **Features** -> **MCP Servers** -> **Add New MCP Server**:

* **Name**: `synqra`
* **Type**: `sse` or `http` (for Streamable HTTP)
* **URL**: `http://127.0.0.1:8000/mcp`

Or using stdio:
* **Name**: `synqra`
* **Type**: `command`
* **Command**: `/path/to/synqra/.venv/bin/python -m fastmcp run src/synqra/server/mcp_server.py:mcp_server --transport stdio`

---

## 8. Planned Integrations (Phase 2+)

1. **YouTube**: Search educational videos, tutorials, and lecture transcripts.
2. **Google Drive**: Create, upload, organize, and retrieve documents.
3. **GitHub**: Search code repositories, fetch files, issues, and commit details.
4. **Google Calendar**: Create, schedule, and query calendar events.
5. **Web Search**: Query live web documentation, technical articles, and resources.
6. **Gmail**: Send automated emails, learning summaries, and progress reports.

---

## 9. Quality Verification & Testing

Run the automated test suite:
```bash
uv run pytest
```

Run code formatting and linting checks:
```bash
uv run ruff check .
uv run ruff format --check .
```

