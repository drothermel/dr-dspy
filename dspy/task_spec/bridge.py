"""Temporary bridge from legacy Signature classes to TaskSpec instances."""

from typing import TYPE_CHECKING, cast

from dspy.task_spec.field_spec import FieldSpec
from dspy.task_spec.task_spec import TaskSpec
from dspy.utils.constants import IS_TYPE_UNDEFINED

if TYPE_CHECKING:
    from dspy.signatures.signature import Signature


def task_spec_from_signature(signature: type["Signature"]) -> TaskSpec:
    """Convert a legacy ``type[Signature]`` into a ``TaskSpec`` instance."""
    inputs: list[FieldSpec] = []
    outputs: list[FieldSpec] = []

    for name, field in signature.input_fields.items():
        extra = cast("dict", field.json_schema_extra or {})
        inputs.append(
            FieldSpec.input(
                name,
                field.annotation or str,
                desc=extra.get("desc", f"${{{name}}}"),
                prefix=extra.get("prefix"),
                is_type_undefined=bool(extra.get(IS_TYPE_UNDEFINED, False)),
                constraints=extra.get("constraints"),
            )
        )

    for name, field in signature.output_fields.items():
        extra = cast("dict", field.json_schema_extra or {})
        outputs.append(
            FieldSpec.output(
                name,
                field.annotation or str,
                desc=extra.get("desc", f"${{{name}}}"),
                prefix=extra.get("prefix"),
                constraints=extra.get("constraints"),
            )
        )

    return TaskSpec(
        name=getattr(signature, "__name__", "TaskSpec"),
        instructions=signature.instructions,
        inputs=tuple(inputs),
        outputs=tuple(outputs),
    )
