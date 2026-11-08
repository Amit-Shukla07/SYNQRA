"""Centralized Agent Tool Registry for SYNQRA.

Maintains explicitly registered SYNQRA tools with their canonical Pydantic input schemas,
permission classifications, integration origins, and execution functions.
"""

import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from synqra.agents.exceptions import SynqraAgentError
from synqra.agents.schemas import PermissionLevel

logger = logging.getLogger("synqra.agents.tool_registry")


@dataclass
class RegisteredTool:
    """Represents a discovered and adapted tool exposed to the agentic engine."""

    name: str
    description: str
    integration_name: str
    input_schema: type[BaseModel]
    execution_function: Callable[..., Any]
    permission_level: PermissionLevel
    is_active: bool = True
    is_async: bool = True

    def __post_init__(self) -> None:
        """Inspect execution_function to auto-detect async nature if not explicitly specified."""
        if inspect.iscoroutinefunction(self.execution_function):
            self.is_async = True


class AgentToolRegistry:
    """Central registry governing available tools for agent planning and execution."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register_tool(self, tool: RegisteredTool) -> None:
        """Register an explicitly verified tool with the registry.

        Args:
            tool: RegisteredTool definition.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(
                f"Tool '{tool.name}' is already registered in the agent tool registry."
            )

        self._tools[tool.name] = tool
        logger.debug(
            "Registered agent tool '%s' (integration=%s, permission=%s)",
            tool.name,
            tool.integration_name,
            tool.permission_level.value,
        )

    def unregister_tool(self, name: str) -> bool:
        """Remove a tool from the registry. Returns True if removed, False otherwise."""
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get_tool(self, name: str) -> RegisteredTool | None:
        """Retrieve a registered tool by name, returning None if not found."""
        return self._tools.get(name)

    def get_tool_or_raise(self, name: str) -> RegisteredTool:
        """Retrieve a registered tool by name or raise SynqraAgentError."""
        tool = self.get_tool(name)
        if tool is None:
            raise SynqraAgentError(f"Tool '{name}' is not registered in the agent tool registry.")
        return tool

    def list_tools(self, active_only: bool = True) -> list[RegisteredTool]:
        """List all registered tools, optionally filtering for active tools only."""
        if active_only:
            return [t for t in self._tools.values() if t.is_active]
        return list(self._tools.values())

    def list_tool_names(self, active_only: bool = True) -> list[str]:
        """Return names of registered tools."""
        return [t.name for t in self.list_tools(active_only=active_only)]

    def get_tools_by_integration(
        self,
        integration_name: str,
        active_only: bool = True,
    ) -> list[RegisteredTool]:
        """Retrieve tools originating from a specific integration."""
        return [
            t
            for t in self.list_tools(active_only=active_only)
            if t.integration_name.lower() == integration_name.lower()
        ]

    def get_tools_by_permission(
        self,
        permission_level: PermissionLevel,
        active_only: bool = True,
    ) -> list[RegisteredTool]:
        """Retrieve tools matching a specific permission tier."""
        return [
            t
            for t in self.list_tools(active_only=active_only)
            if t.permission_level == permission_level
        ]

    def get_tool_description(self, name: str) -> str:
        """Retrieve the human-readable description of a tool."""
        return self.get_tool_or_raise(name).description

    def get_tool_schema(self, name: str) -> type[BaseModel]:
        """Retrieve the Pydantic input schema for a tool."""
        return self.get_tool_or_raise(name).input_schema

    def validate_tool_name(self, name: str) -> bool:
        """Verify whether a tool exists in the registry."""
        return name in self._tools

    def enable_tool(self, name: str) -> None:
        """Enable a registered tool."""
        tool = self.get_tool_or_raise(name)
        tool.is_active = True

    def disable_tool(self, name: str) -> None:
        """Disable a registered tool, temporarily blocking agent execution."""
        tool = self.get_tool_or_raise(name)
        tool.is_active = False

    def clear(self) -> None:
        """Clear all registered tools (primarily used in test teardown)."""
        self._tools.clear()
