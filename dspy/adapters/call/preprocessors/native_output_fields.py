from __future__ import annotations

from typing import TYPE_CHECKING, cast

from dspy.adapters.types.citation import Citations
from dspy.adapters.types.field_type import is_field_type_class
from dspy.adapters.types.reasoning import Reasoning

if TYPE_CHECKING:
    from dspy.adapters.call.preprocessors.context import PreprocessState
    from dspy.core.types import LMConfig


class NativeOutputFieldPreprocessor:
    def run(self, state: PreprocessState) -> PreprocessState:
        adapter = state.adapter
        task_spec = state.task_spec
        for name, field in task_spec.output_fields.items():
            field_type = field.type_
            if not (
                isinstance(field_type, type)
                and field_type in adapter.native_response_types
                and is_field_type_class(field_type)
            ):
                continue
            adapter._ensure_native_response_type_parses_output(field_type)
            if field_type is Reasoning:
                task_spec = adapter._adapt_reasoning_native(
                    task_spec=task_spec,
                    field_name=name,
                    lm=state.lm,
                    config=cast("LMConfig", state.config),
                )
            elif field_type is Citations:
                task_spec = adapter._adapt_citations_native(task_spec=task_spec, field_name=name, lm=state.lm)
            else:
                task_spec = task_spec.delete(name)
        state.task_spec = task_spec
        return state
