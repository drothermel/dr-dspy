from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dspy.adapters.prompt_format import translate_field_type
from dspy.task_spec import field_bindings

if TYPE_CHECKING:
    from dspy.task_spec import TaskSpec
    from dspy.task_spec.field_spec import FieldRole

FIELD_STRUCTURE_INTRO = (
    "All interactions will be structured in the following way, with the appropriate values filled in."
)


def build_field_structure_instructions(
    *,
    input_section: str,
    output_section: str,
    input_preamble: str | None = None,
    output_preamble: str | None = None,
    completed_marker: str | None = None,
) -> str:
    parts = [FIELD_STRUCTURE_INTRO]
    if input_preamble:
        parts.append(input_preamble)
    parts.append(input_section)
    if output_preamble:
        parts.append(output_preamble)
    parts.append(output_section)
    if completed_marker:
        parts.append(completed_marker)
    return "\n\n".join(parts).strip()


def build_role_field_sections(
    adapter: Any,
    task_spec: TaskSpec,
    role: FieldRole,
    *,
    role_label: str | None = None,
) -> str:
    fields_with_values = {
        binding: translate_field_type(binding.field) for binding in field_bindings(task_spec, role=role)
    }
    if role_label is not None:
        return adapter.format_field_with_value(fields_with_values=fields_with_values, role=role_label)
    return adapter.format_field_with_value(fields_with_values=fields_with_values)
