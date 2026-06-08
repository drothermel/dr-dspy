from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from dspy.task_spec.task_spec import TaskSpec


@runtime_checkable
class Predictor(Protocol):
    task_spec: TaskSpec
    demos: list[Any]
    lm: Any
    run: Any | None

    def dump_state(self, json_mode: bool = True) -> dict[str, Any]: ...

    def load_state(
        self,
        state: dict[str, Any],
        *,
        allow_unsafe_lm_state: bool = False,
        custom_types: dict[str, type] | None = None,
    ) -> Any: ...
