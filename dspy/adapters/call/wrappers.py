from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dspy.adapters.base import Adapter
from dspy.task_spec import TaskSpec, input_field

if TYPE_CHECKING:
    from collections.abc import Mapping

    from dspy.clients.base_lm import BaseLM
    from dspy.core.types.config import LMConfig
    from dspy.runtime.config import CallSite
    from dspy.runtime.run_context import RunContext


class HintInjectingAdapter(Adapter):
    def __init__(self, inner: Adapter, hint_map: dict[str, str], task_spec_to_name: dict[TaskSpec, str]) -> None:
        super().__init__(
            callbacks=inner.callbacks,
            use_native_function_calling=inner.use_native_function_calling,
            native_response_types=inner.native_response_types,
            parallel_tool_calls=inner.parallel_tool_calls,
        )
        self._inner = inner
        self._hint_map = hint_map
        self._task_spec_to_name = task_spec_to_name
        self.response_format_policy = inner.response_format_policy
        self.parse_fallback_policy = inner.parse_fallback_policy
        self.capabilities = inner.capabilities

    def format(self, task_spec: TaskSpec, demos: list[dict[str, Any]], inputs: dict[str, Any]) -> list[Any]:
        return self._inner.format(task_spec=task_spec, demos=demos, inputs=inputs)

    def parse(self, task_spec: TaskSpec, completion: str) -> dict[str, Any]:
        return self._inner.parse(task_spec=task_spec, completion=completion)

    async def __call__(
        self,
        *,
        lm: BaseLM,
        config: LMConfig | Mapping[str, Any] | None,
        task_spec: TaskSpec,
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
        run: RunContext,
        call_site: CallSite | None = None,
    ) -> list[dict[str, Any]]:
        from dspy.adapters.call.pipeline import AdapterCallPipeline

        hint_name = self._task_spec_to_name.get(task_spec, "N/A")
        inputs = dict(inputs)
        inputs["hint_"] = self._hint_map.get(hint_name, "N/A")
        hinted_task_spec = task_spec.append(
            input_field("hint_", str, desc="A hint to the module from an earlier run"),
        )
        return await AdapterCallPipeline.execute(
            self._inner,
            lm=lm,
            config=config,
            task_spec=hinted_task_spec,
            demos=demos,
            inputs=inputs,
            run=run,
            call_site=call_site,
        )
