"""TaskSpec shape invariants (field names, roles, non-empty fields).

Distinct from ``validation.py``, which validates runtime task *inputs* at call sites.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dspy.task_spec.field_spec import FieldSpec, validate_field_name

if TYPE_CHECKING:
    from dspy.task_spec.task_spec import TaskSpec


def validate_non_empty_fields(
    inputs: tuple[FieldSpec, ...],
    outputs: tuple[FieldSpec, ...],
) -> None:
    if not inputs and not outputs:
        raise ValueError("TaskSpec must have at least one input or output field.")


def validate_task_spec_field_names(
    inputs: tuple[FieldSpec, ...],
    outputs: tuple[FieldSpec, ...],
) -> None:
    def _check_within_role_duplicates(fields: tuple[FieldSpec, ...], role_label: str) -> None:
        names = [field.name for field in fields]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            quoted = ", ".join(f"'{name}'" for name in duplicates)
            raise ValueError(f"Duplicate {role_label} field name(s): {quoted}.")

    _check_within_role_duplicates(inputs, "input")
    _check_within_role_duplicates(outputs, "output")
    for field in (*inputs, *outputs):
        validate_field_name(field.name)
    cross_role = sorted({field.name for field in inputs}.intersection(field.name for field in outputs))
    if cross_role:
        raise ValueError(
            f"Input and output fields must have distinct names, but found duplicates: '{', '.join(cross_role)}'."
        )


def validate_task_spec(spec: TaskSpec) -> None:
    """Validate TaskSpec invariants: unique field names within/between roles and at least one field."""
    validate_task_spec_field_names(spec.inputs, spec.outputs)
    validate_non_empty_fields(spec.inputs, spec.outputs)
