from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from dspy.adapters.types.tool import Tool  # noqa: TC001 — runtime validate_tools signature
from dspy.predict.call_options import PredictOptions
from dspy.predict.rlm.sync_bridge import _run_sub_lm_async
from dspy.predict.tools import RLM_RESERVED_TOOL_NAMES, validate_tool_names
from dspy.runtime.config import CallSite

if TYPE_CHECKING:
    from collections.abc import Callable

    from dspy.predict.rlm.module import RLM
    from dspy.runtime.run_context import RunContext


def validate_tools(tools: dict[str, Tool]) -> None:
    validate_tool_names(
        tools,
        reserved=RLM_RESERVED_TOOL_NAMES,
        require_identifiers=True,
    )


def format_tool_docs(tools: dict[str, Tool]) -> str:
    if not tools:
        return ""
    lines = ["\nAdditional tools available (use these instead of standard library equivalents):"]
    for tool in tools.values():
        params = []
        for arg_name, arg_schema in (tool.args or {}).items():
            arg_type = arg_schema.get("type", "Any")
            params.append(f"{arg_name}: {arg_type}")
        params_str = ", ".join(params)
        sig_str = f"{tool.name}({params_str})"
        desc = (tool.desc or "No description").replace("\n", "  ")
        lines.append(f"- `{sig_str}` - {desc}")
    return "\n".join(lines)


def make_llm_tools(rlm: RLM, run: RunContext | None = None, max_workers: int = 8) -> dict[str, Callable]:
    state = {"call_count": 0}
    lock = threading.Lock()
    lm = rlm.sub_lm

    def _check_and_increment(n: int = 1) -> None:
        with lock:
            if state["call_count"] + n > rlm.max_llm_calls:
                raise RuntimeError(
                    f"LLM call limit exceeded: {state['call_count']} + {n} > {rlm.max_llm_calls}. Use Python code for aggregation instead of making more LLM calls."
                )
            state["call_count"] += n

    async def _aquery_lm(prompt: str) -> str:
        target_lm = lm if lm is not None else (run.lm if run is not None else None)
        if target_lm is None:
            raise RuntimeError(
                "No LM configured. Pass run=RunContext.create(lm=..., adapter=...) or pass sub_lm to RLM."
            )
        sub_run = (
            run.fork(
                lm=target_lm,
                call_site=CallSite(module="RLM", phase="rlm.sub_lm", lm_role="sub_lm"),
            )
            if run is not None
            else None
        )
        prediction = await rlm._sub_query_predict(
            prompt=prompt,
            run=sub_run,
            options=PredictOptions(lm=target_lm),
        )
        return prediction.response

    def _query_lm(prompt: str) -> str:
        return _run_sub_lm_async(_aquery_lm(prompt))

    def llm_query(prompt: str) -> str:
        if not prompt:
            raise ValueError("prompt cannot be empty")
        _check_and_increment(1)
        return _query_lm(prompt)

    def llm_query_batched(prompts: list[str]) -> list[str]:
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
