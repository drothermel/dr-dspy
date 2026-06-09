from __future__ import annotations

from typing import TYPE_CHECKING

from dspy.adapters.call.postprocess import strip_native_response_output_fields
from dspy.task_spec import TaskSpec, input_field, make_task_spec

if TYPE_CHECKING:
    from dspy.adapters.types.field_type import NativeResponseFieldType


def build_extractor_task_spec(
    original_task_spec: TaskSpec,
    *,
    native_response_types: list[type[NativeResponseFieldType]],
) -> TaskSpec:
    extractable_spec = strip_native_response_output_fields(original_task_spec, native_response_types)
    new_fields = {
        "text": input_field(
            "text", str, desc="Raw completion text from the main language model to extract structured fields from."
        ),
        **dict(extractable_spec.output_fields),
    }
    outputs_str = ", ".join(f"`{field}`" for field in extractable_spec.output_fields)
    instructions = f"The input is a text that should contain all the necessary information to produce the fields {outputs_str}. Your job is to extract the fields from the text verbatim. Extract precisely the appropriate value (content) for each field."
    return make_task_spec(new_fields, instructions=instructions, name="framework.two_step.extractor")
