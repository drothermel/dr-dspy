from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from dspy.adapters.types.tool import Tool, tool_from_callable
from dspy.dsp.utils.settings import settings
from dspy.predict.rlm.sync_bridge import _run_sub_lm_async
from dspy.utils.transparency import reset_active_call_metadata, set_active_call_metadata

if TYPE_CHECKING:
    from collections.abc import Callable

    from dspy.predict.rlm.module import RLM

# Reserved tool names that conflict with built-in sandbox functions
RESERVED_TOOL_NAMES = frozenset({"llm_query", "llm_query_batched", "SUBMIT", "print"})


def normalize_tools(tools: list[Callable] | None) -> dict[str, Tool]:
    """Normalize tools list to a dict of Tool objects keyed by name."""
    if not tools:
        return {}

    if isinstance(tools, dict):
        raise TypeError(
            "tools must be a list, not a dict. "
            "Change tools={'name': func} to tools=[func] "
            "(tool names are inferred from function names, or use Tool(func, name='custom_name'))"
        )

    def to_tool(func: Callable | Tool) -> Tool:
        if isinstance(func, Tool):
            return func
        if not callable(func):
            raise TypeError(f"Tool {func!r} must be callable, got {type(func).__name__}")
        return tool_from_callable(func)

    # List of callables/Tools -> normalize to Tool objects
    tool_list = [to_tool(t) for t in tools]
    tool_map: dict[str, Tool] = {}
    for tool in tool_list:
        if tool.name is None:
            raise ValueError("Tool name could not be determined.")
        tool_map[tool.name] = tool
    return tool_map


def validate_tools(tools: dict[str, Tool]) -> None:
    """Validate user-provided tools have valid names."""
    for name in tools:
        if not name.isidentifier():
            raise ValueError(f"Invalid tool name '{name}': must be a valid Python identifier")
        if name in RESERVED_TOOL_NAMES:
            raise ValueError(f"Tool name '{name}' conflicts with built-in sandbox function")


def format_tool_docs(tools: dict[str, Tool]) -> str:
    """Format user-provided tools for inclusion in instructions."""
    if not tools:
        return ""

    lines = ["\nAdditional tools available (use these instead of standard library equivalents):"]
    for tool in tools.values():
        # Build signature string from Tool's args
        params = []
        for arg_name, arg_schema in (tool.args or {}).items():
            arg_type = arg_schema.get("type", "Any")
            params.append(f"{arg_name}: {arg_type}")
        params_str = ", ".join(params)
        sig_str = f"{tool.name}({params_str})"

        # Get description with newlines escaped
        desc = (tool.desc or "No description").replace("\n", "  ")
        lines.append(f"- `{sig_str}` - {desc}")

    return "\n".join(lines)


def make_llm_tools(rlm: RLM, max_workers: int = 8) -> dict[str, Callable]:
    """Create llm_query and llm_query_batched tools with a fresh call counter."""
    state = {"call_count": 0}
    lock = threading.Lock()
    lm = rlm.sub_lm

    def _check_and_increment(n: int = 1) -> None:
        with lock:
            if state["call_count"] + n > rlm.max_llm_calls:
                raise RuntimeError(
                    f"LLM call limit exceeded: {state['call_count']} + {n} > {rlm.max_llm_calls}. "
                    f"Use Python code for aggregation instead of making more LLM calls."
                )
            state["call_count"] += n

    async def _aquery_lm(prompt: str) -> str:
        target_lm = lm if lm is not None else settings.lm
        if target_lm is None:
            raise RuntimeError(
                "No LM configured. Use `from dspy.dsp.utils.settings import settings; "
                "settings.configure(lm=...)` or pass sub_lm to RLM."
            )
        metadata_token = set_active_call_metadata(
            module="RLM",
            phase="rlm.sub_lm",
            lm_role="sub_lm",
        )
        try:
            prediction = await rlm._sub_query_predict(prompt=prompt, lm=target_lm)
        finally:
            reset_active_call_metadata(metadata_token)
        return prediction.response

    def _query_lm(prompt: str) -> str:
        return _run_sub_lm_async(_aquery_lm(prompt))

    def llm_query(prompt: str) -> str:
        """Query the LLM with a prompt string."""
        if not prompt:
            raise ValueError("prompt cannot be empty")
        _check_and_increment(1)
        return _query_lm(prompt)

    def llm_query_batched(prompts: list[str]) -> list[str]:
        """Query the LLM with multiple prompts concurrently."""
        if not prompts:
            return []
        _check_and_increment(len(prompts))

        results: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(_query_lm, p): i for i, p in enumerate(prompts)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = f"[ERROR] {e}"
        return [results[i] for i in range(len(prompts))]

    return {"llm_query": llm_query, "llm_query_batched": llm_query_batched}
