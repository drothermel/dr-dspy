from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

from dspy.utils.exceptions import AdapterParseError

if TYPE_CHECKING:
    from dspy.adapters.base.adapter import Adapter
    from dspy.clients.base_lm import BaseLM
    from dspy.core.types.config import LMConfig
    from dspy.task_spec import TaskSpec


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
        error: AdapterParseError,
    ) -> list[dict[str, Any]]:
        _ = (adapter, lm, config, task_spec, demos, inputs)
        raise error


class JSONParseFallbackPolicy:
    def __init__(self, fallback_factory: Callable[[], Adapter]) -> None:
        self._fallback_factory = fallback_factory

    async def execute_fallback(
        self,
        *,
        adapter: Adapter,
        lm: BaseLM,
        config: LMConfig | None,
        task_spec: TaskSpec,
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
        error: AdapterParseError,
    ) -> list[dict[str, Any]]:
        _ = (adapter, error)
        fallback = self._fallback_factory()
        from dspy.adapters.call.pipeline import AdapterCallPipeline

        return await AdapterCallPipeline.execute(
            fallback,
            lm=lm,
            config=config,
            task_spec=task_spec,
            demos=demos,
            inputs=inputs,
            allow_parse_fallback=False,
        )
