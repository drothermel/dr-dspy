"""Factory for constructing TaskSpec instances."""

from collections.abc import Sequence

from dspy.task_spec.field_spec import FieldRole, FieldSpec
from dspy.task_spec.parse import parse_task_spec_string
from dspy.task_spec.task_spec import TaskSpec


def make_task_spec(
    spec: str | dict[str, FieldSpec] | None = None,
    *,
    inputs: Sequence[FieldSpec] | None = None,
    outputs: Sequence[FieldSpec] | None = None,
    instructions: str,
    name: str | None = None,
    custom_types: dict[str, type] | None = None,
) -> TaskSpec:
    """Create a TaskSpec instance. This is the only supported public constructor.

    Args:
        spec: Either a string ``"input1, input2 -> output"`` or a dict mapping field
            names to ``FieldSpec`` instances.
        inputs: Optional sequence of input ``FieldSpec`` instances.
        outputs: Optional sequence of output ``FieldSpec`` instances.
        instructions: Required task instructions sent to the language model.
        name: Optional name for the task spec. Defaults to ``TaskSpec`` for string specs
            or the first field-derived name for dict specs.
        custom_types: Optional mapping of type names to type objects for string parsing.

    Returns:
        A frozen ``TaskSpec`` instance.
    """
    if not instructions:
        raise ValueError("instructions is required and must be non-empty.")

    has_spec = spec is not None
    has_field_lists = inputs is not None or outputs is not None
    if has_spec and has_field_lists:
        raise TypeError("Pass either spec or inputs/outputs, not both.")
    if not has_spec and not has_field_lists:
        raise TypeError("One of spec or inputs/outputs is required.")

    if isinstance(spec, str):
        parsed_inputs, parsed_outputs = parse_task_spec_string(spec, custom_types=custom_types)
        resolved_name = name or "TaskSpec"
        return TaskSpec(
            name=resolved_name,
            instructions=instructions,
            inputs=parsed_inputs,
            outputs=parsed_outputs,
        )

    if has_field_lists:
        input_fields = tuple(inputs or ())
        output_fields = tuple(outputs or ())
        _validate_field_roles(input_fields, FieldRole.INPUT)
        _validate_field_roles(output_fields, FieldRole.OUTPUT)
        if not input_fields and not output_fields:
            raise ValueError("inputs and outputs must contain at least one field.")
        resolved_name = name or (input_fields[0].name if input_fields else output_fields[0].name)
        return TaskSpec(
            name=resolved_name,
            instructions=instructions,
            inputs=input_fields,
            outputs=output_fields,
        )

    if not isinstance(spec, dict):
        raise TypeError(f"spec must be str or dict[str, FieldSpec], got {type(spec).__name__}.")

    input_fields: list[FieldSpec] = []
    output_fields: list[FieldSpec] = []
    for field_name, field in spec.items():
        if field.name != field_name:
            raise ValueError(f"Field dict key {field_name!r} does not match FieldSpec.name {field.name!r}.")
        if field.role == FieldRole.INPUT:
            input_fields.append(field)
        else:
            output_fields.append(field)

    if not input_fields and not output_fields:
        raise ValueError("spec dict must contain at least one input or output field.")

    resolved_name = name or (input_fields[0].name if input_fields else output_fields[0].name)
    return TaskSpec(
        name=resolved_name,
        instructions=instructions,
        inputs=tuple(input_fields),
        outputs=tuple(output_fields),
    )


def _validate_field_roles(fields: tuple[FieldSpec, ...], expected_role: FieldRole) -> None:
    for field in fields:
        if field.role != expected_role:
            raise ValueError(f"Field {field.name!r} has role {field.role!r}, expected {expected_role.value!r}.")
