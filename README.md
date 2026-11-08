# SYNQRA — Universal MCP Integration Server & AI Agent Orchestrator

SYNQRA is a production-grade integration platform connecting AI agents with external productivity and developer services through the **Model Context Protocol (MCP)** and a deterministic **LangGraph agent orchestration engine**.

---

## 🌟 Key Capabilities

- **31 Curated Integration Tools**: Curated tool definitions for YouTube, GitHub, Google Drive / Docs, Google Calendar, Gmail, and Web Search (Tavily).
- **Additive Agent MCP Tool (`synqra_run_agent`)**: Exposes multi-step autonomous workflows directly to MCP clients (Claude, Cursor, Antigravity).
- **Human-in-the-Loop (HITL) Security**: Automated execution for safe read-only operations, with strict cryptographic approval interruption for `WRITE` and `SENSITIVE` operations.
- **Durable Persistence & Recovery**: PostgreSQL-backed approval repository and LangGraph checkpointing (`AsyncPostgresSaver`) allowing interrupted workflows to survive process restarts without duplicate side-effects.
- **Unified CLI & Interactive Chat**: Single-command execution, multi-turn conversational chat, and system diagnostics.

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/your-org/synqra.git
cd synqra
uv sync
```

### Environment Configuration

Copy the example environment configuration:

```bash
cp .env.example .env
```

Configure your credentials in `.env`:

```ini
# Core Server Configuration
SYNQRA_APP_NAME="SYNQRA"
SYNQRA_PORT=8000
SYNQRA_HOST="127.0.0.1"

# Persistence & Checkpointing ("memory" for local dev, "postgres" for production)
SYNQRA_CHECKPOINTER_TYPE="postgres"
SYNQRA_DATABASE_URL="postgresql+psycopg://user:password@localhost:5432/synqra"

# Approval Security Secret Key (Required for signing approval tokens)
SYNQRA_APPROVAL_SECRET_KEY="your-high-entropy-approval-secret-key"
SYNQRA_APPROVAL_API_KEY="your-primary-operator-bearer-token"

# Integration API Keys
COMPOSIO_API_KEY="your-composio-api-key"
TAVILY_API_KEY="your-tavily-api-key"
OPENAI_API_KEY="your-openai-api-key"
```

---

## 💻 CLI Usage

SYNQRA provides a unified command-line interface:

### 1. Help & Diagnostics

```bash
# View all commands
synqra --help

# Display version and environment information
synqra version

# Inspect system status and connected integration readiness
synqra status
```

### 2. Single-Command Agent Workflow (`synqra run`)

Execute a natural-language task end-to-end:

```bash
synqra run "Search GitHub for popular FastAPI projects and summarize them"
```

If a step involves a write operation (e.g., sending an email or creating a calendar event), execution pauses safely and renders the approval requirements:

```
╔════════════════════════════════════════════════════════════════════════════╗
║                   SECURITY INTERRUPT (Human-in-the-Loop)                   ║
╚════════════════════════════════════════════════════════════════════════════╝
⚠️ Human Authorization Required
This workflow requested a protected WRITE operation:
• Approval Request ID: appr_9a1f2c3d
• Tool Name:           send_gmail_message
• Workflow ID:         wf_7b8c9d0e
• Thread ID:           th_1a2b3c4d
• Action Purpose:      Send summary notification
To authorize this operation:
  POST /api/approvals/appr_9a1f2c3d/approve
  Header: Authorization: Bearer <operator-token>
```

### 3. Interactive Terminal Chat (`synqra chat`)

Launch a multi-turn terminal session with conversation context:

```bash
synqra chat
```

```text
╔══════════════════════════════════════════════╗
║                  SYNQRA                      ║
║          Universal AI Integration Agent      ║
╚══════════════════════════════════════════════╝

Connected integrations:
 ✓ GitHub               Ready
 ✓ Gmail                Ready
 ✓ Google Drive         Ready
 ✓ Google Calendar      Ready
 ✓ YouTube              Ready
 ✓ Web Search           Ready

SYNQRA > Find my latest GitHub repositories.
[1] Searching GitHub repositories...
SYNQRA > Here are your latest repositories: ...

SYNQRA > Summarize the last repository for me.
[1] Retrieving repository README...
SYNQRA > The repository is ...
```

Interactive commands during chat:
- `/help` — Display available commands
- `/status` — View integration and checkpointer readiness
- `/clear` — Reset session context and thread ID
- `/exit` — Cleanly exit the chat session

### 4. Running the MCP Server

Start the FastMCP Streamable HTTP server:

```bash
synqra server --host 127.0.0.1 --port 8000
# or direct script
synqra-server
```

---

## 🤖 MCP Integration & Agent Tool

### Curated Tools (31)

SYNQRA exposes 31 curated integration tools:
- **YouTube** (2): `search_youtube_videos`, `get_video_details`
- **GitHub** (5): `search_github_repositories`, `get_github_repository`, `get_github_file_contents`, `list_github_issues`, `get_github_readme`
- **Google Drive / Docs** (7): `search_drive_files`, `get_drive_file`, `download_drive_file`, `upload_drive_file`, `create_google_document`, `update_google_document`, `create_drive_folder`
- **Google Calendar** (7): `list_calendars`, `list_calendar_events`, `get_calendar_event`, `create_calendar_event`, `update_calendar_event`, `delete_calendar_event`, `find_free_time`
- **Gmail** (6): `search_gmail_messages`, `get_gmail_message`, `list_gmail_labels`, `create_gmail_draft`, `send_gmail_message`, `reply_to_gmail_message`
- **Web Search** (3): `search_web`, `extract_webpage_content`, `search_news`
- **System Diagnostics** (1): `health_check`

### Agent MCP Tool (`synqra_run_agent`)

In addition to individual tools, AI clients can delegate full orchestration tasks to the SYNQRA agent:

```json
{
  "name": "synqra_run_agent",
  "arguments": {
    "request": "Find my upcoming calendar events for tomorrow and draft an email summary to team@example.com"
  }
}
```

#### Response Structure:

```json
{
  "workflow_id": "wf_abc123",
  "thread_id": "th_xyz789",
  "status": "waiting_for_approval",
  "message": "Workflow paused awaiting human approval for protected operation 'create_gmail_draft'.",
  "steps": [
    {
      "step_id": "step_1",
      "tool_name": "list_calendar_events",
      "description": "Fetch upcoming calendar events",
      "status": "completed"
    },
    {
      "step_id": "step_2",
      "tool_name": "create_gmail_draft",
      "description": "Create draft email summary",
      "status": "awaiting_approval"
    }
  ],
  "approval_required": true,
  "pending_approvals": [
    {
      "approval_id": "appr_456def",
      "tool_name": "create_gmail_draft",
      "arguments": {
        "recipient": "team@example.com",
        "subject": "Tomorrow's Schedule",
        "body": "..."
      },
      "permission_level": "write",
      "reason": "Drafting schedule email"
    }
  ],
  "result": {}
}
```

---

## 🔒 Security Architecture

1. **Deterministic Permission Matrix**:
   - `READ_ONLY`: Automated execution.
   - `WRITE` & `SENSITIVE`: Always interrupted via LangGraph checkpointer and awaiting approval.
2. **Cryptographic Anti-Tampering**:
   - Every approval request binds tool arguments with an HMAC-SHA256 signature.
   - Resumption validates hash integrity before execution.
3. **Atomic Consumption (CAS)**:
   - Approval records are single-use (`UPDATE approval_requests SET status = 'consumed' WHERE status = 'approved'`).
   - Ambiguous execution outcomes fail closed and never duplicate write operations.
4. **Role-Based Access Control (RBAC)**:
   - Workflow-level operator isolation (`allowed_workflows`).
   - Constant-time bearer token authentication.

---

## 🧪 Testing

Run the full pytest suite:

```bash
uv run pytest -v
```

Run code formatting and lint checks:

```bash
uv run ruff format --check .
uv run ruff check .
```

