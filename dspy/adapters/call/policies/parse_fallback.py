from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping

    from dspy.adapters.base.adapter import Adapter
    from dspy.clients.base_lm import BaseLM
    from dspy.core.types import LMConfig
    from dspy.errors import AdapterParseError
    from dspy.runtime.config import CallSite
    from dspy.runtime.run_context import RunContext
    from dspy.task_spec import TaskSpec


class PipelineExecutor(Protocol):
    async def __call__(
        self,
        adapter: Adapter,
        *,
        lm: BaseLM,
        config: LMConfig | Mapping[str, Any] | None,
        task_spec: TaskSpec,
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
        run: RunContext,
        call_site: CallSite | None = None,
        allow_parse_fallback: bool = True,
    ) -> list[dict[str, Any]]: ...


class ParseFallbackPolicy(Protocol):
    async def execute_fallback(
        self,
        *,
        adapter: Adapter,
        lm: BaseLM,
        config: LMConfig | None,
        task_spec: TaskSpec,
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
        run: RunContext,
        error: AdapterParseError,
    ) -> list[dict[str, Any]]: ...


class NoOpParseFallbackPolicy:
    async def execute_fallback(
        self,
        *,
        adapter: Adapter,
        lm: BaseLM,
        config: LMConfig | None,
        task_spec: TaskSpec,
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
        run: RunContext,
        error: AdapterParseError,
    ) -> list[dict[str, Any]]:
        _ = (adapter, lm, config, task_spec, demos, inputs, run)
        raise error
