"""Temporary bridge between legacy Signature classes and TaskSpec instances."""

from typing import TYPE_CHECKING, cast

from dspy.task_spec.field_spec import FieldSpec
from dspy.task_spec.task_spec import TaskSpec
from dspy.utils.constants import IS_TYPE_UNDEFINED

if TYPE_CHECKING:
    from dspy.signatures.signature import Signature


def _field_spec_to_signature_field(field: FieldSpec):
    from dspy.signatures.field import InputField, OutputField

    kwargs: dict = {}
    if field.desc != f"${{{field.name}}}":
        kwargs["desc"] = field.desc
    if field.constraints:
        kwargs["constraints"] = field.constraints
    field_cls = InputField if field.role == "input" else OutputField
    return field.type_, field_cls(**kwargs)


def signature_from_task_spec(spec: TaskSpec) -> type["Signature"]:
    """Convert a TaskSpec instance into a legacy ``type[Signature]`` for unmigrated callers."""
    from dspy.signatures.signature import make_signature

    fields = {name: _field_spec_to_signature_field(field) for name, field in spec.fields.items()}
    return make_signature(fields, instructions=spec.instructions, signature_name=spec.name)


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
