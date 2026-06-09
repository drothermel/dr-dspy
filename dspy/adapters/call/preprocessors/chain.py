from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from dspy.adapters.call.preprocessors.coerce_config import CoerceConfigPreprocessor
from dspy.adapters.call.preprocessors.context import PreprocessState
from dspy.adapters.call.preprocessors.native_function_calling import NativeFunctionCallingPreprocessor
from dspy.adapters.call.preprocessors.native_output_fields import NativeOutputFieldPreprocessor

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from dspy.adapters.base.protocols import ComposedAdapterT
    from dspy.adapters.call.preprocessors.base import CallPreprocessor
    from dspy.clients.base_lm import BaseLM
    from dspy.core.types import LMConfig, LMToolSpec
    from dspy.task_spec import TaskSpec


class CallPreprocessorChain:
    def __init__(self, preprocessors: Sequence[CallPreprocessor]) -> None:
        self._preprocessors = tuple(preprocessors)

    def run(
        self,
        adapter: ComposedAdapterT,
        *,
        lm: BaseLM,
        config: LMConfig | Mapping[str, Any],
        task_spec: TaskSpec,
        inputs: dict[str, Any],
    ) -> tuple[TaskSpec, list[LMToolSpec], LMConfig]:
        state = PreprocessState(
            adapter=adapter,
            lm=lm,
            config=config,
            task_spec=task_spec,
            inputs=inputs,
        )
        for preprocessor in self._preprocessors:
            state = preprocessor.run(state)
        resolved_config = cast("LMConfig", state.config)
        return state.task_spec, state.tools, resolved_config


def default_preprocessor_chain() -> CallPreprocessorChain:
    return CallPreprocessorChain(
        (
            CoerceConfigPreprocessor(),
            NativeFunctionCallingPreprocessor(),
            NativeOutputFieldPreprocessor(),
        )
    )
