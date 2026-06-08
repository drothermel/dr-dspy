"""Factory for constructing TaskSpec instances."""

from dspy.task_spec.field_spec import FieldSpec
from dspy.task_spec.parse import parse_task_spec_string
from dspy.task_spec.task_spec import TaskSpec


def make_task_spec(
    spec: str | dict[str, FieldSpec],
    *,
    instructions: str,
    name: str | None = None,
    custom_types: dict[str, type] | None = None,
) -> TaskSpec:
    """Create a TaskSpec instance. This is the only supported public constructor.

    Args:
        spec: Either a string ``"input1, input2 -> output"`` or a dict mapping field
            names to ``FieldSpec`` instances.
        instructions: Required task instructions sent to the language model.
        name: Optional name for the task spec. Defaults to ``TaskSpec`` for string specs
            or the first field-derived name for dict specs.
        custom_types: Optional mapping of type names to type objects for string parsing.

    Returns:
        A frozen ``TaskSpec`` instance.
    """
    if not instructions:
        raise ValueError("instructions is required and must be non-empty.")

    if isinstance(spec, str):
        inputs, outputs = parse_task_spec_string(spec, custom_types=custom_types)
        resolved_name = name or "TaskSpec"
        return TaskSpec(name=resolved_name, instructions=instructions, inputs=inputs, outputs=outputs)

    if not isinstance(spec, dict):
        raise TypeError(f"spec must be str or dict[str, FieldSpec], got {type(spec).__name__}.")

    inputs: list[FieldSpec] = []
    outputs: list[FieldSpec] = []
    for field_name, field in spec.items():
        if field.name != field_name:
            raise ValueError(f"Field dict key {field_name!r} does not match FieldSpec.name {field.name!r}.")
        if field.role == "input":
            inputs.append(field)
        else:
            outputs.append(field)

    if not inputs and not outputs:
        raise ValueError("spec dict must contain at least one input or output field.")

    resolved_name = name or (inputs[0].name if inputs else outputs[0].name)
    return TaskSpec(
        name=resolved_name,
        instructions=instructions,
        inputs=tuple(inputs),
        outputs=tuple(outputs),
    )
