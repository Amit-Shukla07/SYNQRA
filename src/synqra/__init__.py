"""SYNQRA – The Universal MCP Integration Server."""

from synqra.core.config import Settings, get_settings
from synqra.server.app import main
from synqra.server.mcp_server import create_mcp_server, mcp_server

__version__ = "0.1.0"
__all__ = [
    "Settings",
    "create_mcp_server",
    "get_settings",
    "main",
    "mcp_server",
]
