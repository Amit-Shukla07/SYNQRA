"""Tool Adapter bridging SYNQRA integration functions to LangChain and LangGraph.

Wraps registered integration tools into LangChain-compatible StructuredTool instances
and provides a unified asynchronous execution boundary with argument validation,
timeout enforcement, and standardized error normalization.
"""

import asyncio
import logging
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import ValidationError

from synqra.agents.tool_registry import RegisteredTool
from synqra.integrations.exceptions import (
    IntegrationExecutionError,
    SynqraIntegrationError,
)

logger = logging.getLogger("synqra.agents.tool_adapter")

DEFAULT_TOOL_TIMEOUT_SECONDS: float = 60.0


class AgentToolAdapter:
    """Adapter bridging SYNQRA RegisteredTool definitions into callable execution units."""

    def __init__(self, default_timeout: float = DEFAULT_TOOL_TIMEOUT_SECONDS) -> None:
        self.default_timeout = default_timeout

    async def execute(
        self,
        tool: RegisteredTool,
        arguments: dict[str, Any],
        timeout: float | None = None,
    ) -> Any:
        """Execute a tool with argument validation, timeout enforcement, and error normalization.

        Args:
            tool: RegisteredTool definition.
            arguments: Input arguments dictionary.
            timeout: Optional execution timeout in seconds (defaults to self.default_timeout).

        Returns:
            The raw structured return value from the underlying integration function.

        Raises:
            IntegrationExecutionError: If validation, timeout, or execution fails.
            SynqraIntegrationError: Preserved from underlying integration if already typed.
        """
        effective_timeout = timeout if timeout is not None else self.default_timeout

        # 1. Validate arguments against canonical Pydantic input schema
        try:
            validated_model = tool.input_schema.model_validate(arguments)
            # Use model_dump to extract validated parameters as dictionary kwargs
            call_kwargs = validated_model.model_dump()
        except ValidationError as exc:
            logger.warning("Argument validation failed for tool '%s': %s", tool.name, exc)
            raise IntegrationExecutionError(
                f"Invalid arguments for tool '{tool.name}': {exc}",
                integration_name=tool.integration_name,
            ) from exc

        # 2. Execute underlying function with timeout
        try:
            if tool.is_async:
                result = await asyncio.wait_for(
                    tool.execution_function(**call_kwargs),
                    timeout=effective_timeout,
                )
            else:
                # Run synchronous function in worker thread to prevent event loop blocking
                result = await asyncio.wait_for(
                    asyncio.to_thread(tool.execution_function, **call_kwargs),
                    timeout=effective_timeout,
                )
            return result

        except TimeoutError as exc:
            logger.error("Tool '%s' timed out after %.1f seconds", tool.name, effective_timeout)
            raise IntegrationExecutionError(
                f"Tool '{tool.name}' execution timed out after {effective_timeout:.1f} seconds.",
                integration_name=tool.integration_name,
            ) from exc

        except SynqraIntegrationError:
            # Preserve existing typed integration exceptions
            raise

        except Exception as exc:
            logger.error("Unhandled error executing tool '%s': %s", tool.name, exc)
            raise IntegrationExecutionError(
                f"Error executing tool '{tool.name}': {exc}",
                integration_name=tool.integration_name,
            ) from exc

    def to_structured_tool(
        self,
        tool: RegisteredTool,
        timeout: float | None = None,
    ) -> StructuredTool:
        """Convert a RegisteredTool into a LangChain StructuredTool instance.

        Args:
            tool: RegisteredTool definition.
            timeout: Optional timeout for execution.

        Returns:
            A LangChain StructuredTool configured with schema and description.
        """
        adapter = self

        async def _coro_runner(**kwargs: Any) -> Any:
            return await adapter.execute(tool=tool, arguments=kwargs, timeout=timeout)

        def _sync_runner(**kwargs: Any) -> Any:
            # Sync runner for environments calling synchronous invoke
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # If already in an event loop, run in a separate thread
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(
                        asyncio.run,
                        adapter.execute(tool=tool, arguments=kwargs, timeout=timeout),
                    )
                    return future.result()
            else:
                return asyncio.run(adapter.execute(tool=tool, arguments=kwargs, timeout=timeout))

        return StructuredTool.from_function(
            func=_sync_runner,
            coroutine=_coro_runner if tool.is_async else None,
            name=tool.name,
            description=tool.description,
            args_schema=tool.input_schema,
        )
