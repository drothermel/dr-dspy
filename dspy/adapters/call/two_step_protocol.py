from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeGuard

from dspy.adapters.call.mode import AdapterCallMode

if TYPE_CHECKING:
    from dspy.adapters.base.adapter import Adapter
    from dspy.runtime.run_context import RunContext
    from dspy.task_spec import TaskSpec


class TwoStepMainAdapter(Protocol):
    call_mode: AdapterCallMode

    def _create_extractor_task_spec(self, original_task_spec: TaskSpec) -> TaskSpec: ...

    async def _run_extraction(self, *, original_task_spec: TaskSpec, text: str, run: RunContext) -> dict[str, Any]: ...

    def _get_tool_call_output_field_name(self, task_spec: TaskSpec) -> str | None: ...


def is_two_step_adapter(adapter: Adapter) -> TypeGuard[TwoStepMainAdapter]:
    return adapter.call_mode == AdapterCallMode.TWO_STEP
