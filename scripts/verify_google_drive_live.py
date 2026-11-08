"""Live end-to-end verification script for SYNQRA Google Drive integration.

Executes JSON-RPC 2.0 calls over Streamable HTTP against http://127.0.0.1:8035/mcp
to verify MCP initialization, tool discovery, health check, and all 7 Google Drive tools.
"""

import json
from typing import Any

import httpx

MCP_URL = "http://127.0.0.1:8035/mcp"


def call_mcp(
    client: httpx.Client,
    method: str,
    params: dict[str, Any] | None = None,
    msg_id: int = 1,
    headers: dict[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    req_headers = {"Accept": "application/json, text/event-stream"}
    if headers:
        req_headers.update(headers)

    payload = {"jsonrpc": "2.0", "id": msg_id, "method": method}
    if params is not None:
        payload["params"] = params

    response = client.post(MCP_URL, json=payload, headers=req_headers, timeout=60.0)
    response.raise_for_status()

    # Handle streamable HTTP response (SSE format or direct JSON)
    resp_text = response.text.strip()
    lines = [line for line in resp_text.splitlines() if line.startswith("data: ")]
    data = json.loads(lines[0][6:]) if lines else response.json()

    resp_headers = dict(response.headers)
    return data, resp_headers


def main() -> None:
    print("=" * 70)
    print("SYNQRA Phase 5: Google Drive Live End-to-End Verification")
    print("=" * 70)

    client = httpx.Client(timeout=60.0)
    active_headers: dict[str, str] = {}

    # 1. MCP Initialize
    print("\n[1] Testing MCP initialize...")
    init_data, resp_hdrs = call_mcp(
        client,
        "initialize",
        {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "synqra-verifier", "version": "1.0"},
        },
        msg_id=1,
    )
    session_id = resp_hdrs.get("mcp-session-id")
    if session_id:
        active_headers["mcp-session-id"] = session_id
        print(f"    Session ID: {session_id}")

    server_info = init_data.get("result", {}).get("serverInfo", {})
    print(f"    Server Info: {server_info.get('name')} v{server_info.get('version')}")

    # 2. Tools List
    print("\n[2] Testing tools/list...")
    tools_data, _ = call_mcp(client, "tools/list", {}, msg_id=2, headers=active_headers)
    tools = tools_data.get("result", {}).get("tools", [])
    tool_names = [t["name"] for t in tools]
    print(f"    Total registered tools: {len(tool_names)}")
    drive_tools = [
        "search_drive_files",
        "get_drive_file",
        "download_drive_file",
        "upload_drive_file",
        "create_google_document",
        "update_google_document",
        "create_drive_folder",
    ]
    for dt in drive_tools:
        status = "✓ FOUND" if dt in tool_names else "✗ MISSING"
        print(f"    - {dt}: {status}")

    # 3. Health Check
    print("\n[3] Testing health_check tool call...")
    health_data, _ = call_mcp(
        client,
        "tools/call",
        {"name": "health_check", "arguments": {}},
        msg_id=3,
        headers=active_headers,
    )
    health_structured = health_data.get("result", {}).get("structuredContent", {})
    drive_health = health_structured.get("integrations_status", {}).get("google_drive", {})
    print(f"    Google Drive State: {drive_health.get('state')}")
    print(f"    Is Connected: {drive_health.get('is_connected')}")
    print(f"    Registered Tools: {drive_health.get('registered_tools')}")

    # 4a. search_drive_files
    print("\n[4a] Testing search_drive_files...")
    search_data, _ = call_mcp(
        client,
        "tools/call",
        {"name": "search_drive_files", "arguments": {"page_size": 5}},
        msg_id=4,
        headers=active_headers,
    )
    print("    Search Result:")
    print(json.dumps(search_data, indent=4))

    # 4b. create_drive_folder: SYNQRA_Live_Test
    print("\n[4b] Testing create_drive_folder ('SYNQRA_Live_Test')...")
    folder_data, _ = call_mcp(
        client,
        "tools/call",
        {"name": "create_drive_folder", "arguments": {"name": "SYNQRA_Live_Test"}},
        msg_id=5,
        headers=active_headers,
    )
    print("    Folder Creation Result:")
    print(json.dumps(folder_data, indent=4))

    # Extract folder ID from folder creation
    content_list = folder_data.get("result", {}).get("content", [])
    created_folder_id = None
    if content_list and content_list[0].get("text"):
        try:
            folder_parsed = json.loads(content_list[0]["text"])
            created_folder_id = folder_parsed.get("folder_id")
        except Exception:
            pass
    print(f"    Created Folder ID: {created_folder_id}")

    # 4c. upload_drive_file: test.txt into SYNQRA_Live_Test
    print("\n[4c] Testing upload_drive_file ('test.txt')...")
    upload_args: dict[str, Any] = {
        "name": "test.txt",
        "content": "Hello from SYNQRA Live Verification!\nTimestamp: 2026-11-07",
        "mime_type": "text/plain",
    }
    if created_folder_id:
        upload_args["parent_id"] = created_folder_id

    upload_data, _ = call_mcp(
        client,
        "tools/call",
        {"name": "upload_drive_file", "arguments": upload_args},
        msg_id=6,
        headers=active_headers,
    )
    print("    Upload Result:")
    print(json.dumps(upload_data, indent=4))

    uploaded_file_id = None
    upload_content_list = upload_data.get("result", {}).get("content", [])
    if upload_content_list and upload_content_list[0].get("text"):
        try:
            upload_parsed = json.loads(upload_content_list[0]["text"])
            uploaded_file_id = upload_parsed.get("id")
        except Exception:
            pass
    print(f"    Uploaded File ID: {uploaded_file_id}")

    # 4d. get_drive_file metadata
    target_file_id = uploaded_file_id
    if target_file_id:
        print(f"\n[4d] Testing get_drive_file on ID: {target_file_id}...")
        get_data, _ = call_mcp(
            client,
            "tools/call",
            {"name": "get_drive_file", "arguments": {"file_id": target_file_id}},
            msg_id=7,
            headers=active_headers,
        )
        print("    Get File Result:")
        print(json.dumps(get_data, indent=4))

    # 4e. download_drive_file on uploaded file
    if target_file_id:
        print(f"\n[4e] Testing download_drive_file on ID: {target_file_id}...")
        download_data, _ = call_mcp(
            client,
            "tools/call",
            {"name": "download_drive_file", "arguments": {"file_id": target_file_id}},
            msg_id=8,
            headers=active_headers,
        )
        print("    Download Result:")
        print(json.dumps(download_data, indent=4))

    # 4f. create_google_document in SYNQRA_Live_Test
    print("\n[4f] Testing create_google_document ('SYNQRA_Test_Doc')...")
    doc_args: dict[str, Any] = {
        "title": "SYNQRA_Test_Doc",
        "content": "This is an automated test document created by SYNQRA MCP server.",
    }
    if created_folder_id:
        doc_args["parent_id"] = created_folder_id

    create_doc_data, _ = call_mcp(
        client,
        "tools/call",
        {"name": "create_google_document", "arguments": doc_args},
        msg_id=9,
        headers=active_headers,
    )
    print("    Create Google Doc Result:")
    print(json.dumps(create_doc_data, indent=4))

    created_doc_id = None
    doc_content_list = create_doc_data.get("result", {}).get("content", [])
    if doc_content_list and doc_content_list[0].get("text"):
        try:
            doc_parsed = json.loads(doc_content_list[0]["text"])
            created_doc_id = doc_parsed.get("document_id")
        except Exception:
            pass
    print(f"    Created Document ID: {created_doc_id}")

    # 4g. update_google_document
    if created_doc_id:
        print(f"\n[4g] Testing update_google_document on ID: {created_doc_id}...")
        update_data, _ = call_mcp(
            client,
            "tools/call",
            {
                "name": "update_google_document",
                "arguments": {
                    "file_id": created_doc_id,
                    "content": ("Updated content: Verification completed successfully."),
                },
            },
            msg_id=10,
            headers=active_headers,
        )
        print("    Update Document Result:")
        print(json.dumps(update_data, indent=4))

    print("\n" + "=" * 70)
    print("Live End-to-End Verification Run Complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
