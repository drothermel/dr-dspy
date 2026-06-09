from __future__ import annotations

import inspect

from dspy.adapters.types.tool import Tool

RLM_RESERVED_TOOL_NAMES = frozenset({"llm_query", "llm_query_batched", "SUBMIT", "print"})


def normalize_tools(
    tools: list[Tool] | None,
    *,
    require_plain_function: bool = False,
) -> dict[str, Tool]:
    if not tools:
        return {}
    if isinstance(tools, dict):
        raise TypeError(
            "tools must be a list, not a dict. Change tools={'name': func} to tools=[Tool(func, description='...')] (tool names are inferred from function names, or use Tool(func, name='custom_name'))"
        )
    tool_map: dict[str, Tool] = {}
    for tool in tools:
        if not isinstance(tool, Tool):
            raise TypeError(
                "tools must be Tool instances with an explicit description. Use Tool(func, description='...')."
            )
        if require_plain_function and not inspect.isfunction(tool.func):
            raise ValueError("CodeAct only accepts functions and not callable objects.")
        if tool.name is None:
            raise ValueError("Tool name could not be determined.")
        tool_map[tool.name] = tool
    return tool_map


def validate_tool_names(
    tools: dict[str, Tool],
    *,
    reserved: frozenset[str] = frozenset(),
    require_identifiers: bool = False,
) -> None:
    for name in tools:
        if require_identifiers and not name.isidentifier():
            raise ValueError(f"Invalid tool name '{name}': must be a valid Python identifier")
        if name in reserved:
            raise ValueError(f"Tool name '{name}' conflicts with built-in sandbox function")
