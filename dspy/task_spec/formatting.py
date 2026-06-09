from dspy.adapters.types.base_type import Type as DspyType
from dspy.adapters.utils.fields import get_annotation_name
from dspy.task_spec.field_spec import FieldSpec


def _format_field_description_lines(
    *,
    entries: list[tuple[str, object, str, str | None]],
) -> str:
    field_descriptions = []
    for idx, (name, annotation, desc, constraints) in enumerate(entries):
        field_message = f"{idx + 1}. `{name}`"
        field_message += f" ({get_annotation_name(annotation)})"
        custom_types = DspyType.extract_custom_type_from_annotation(annotation)
        for custom_type in custom_types:
            if len(custom_type.description()) > 0:
                desc += f"\n    Type description of {get_annotation_name(custom_type)}: {custom_type.description()}"
        field_message += f": {desc}"
        if constraints:
            field_message += f"\nConstraints: {constraints}"
        field_descriptions.append(field_message)
    return "\n".join(field_descriptions).strip()


def get_field_spec_description_string(fields: dict[str, FieldSpec]) -> str:
    entries = [
        (
            name,
            field.type_,
            field.desc if field.desc != f"${{{name}}}" else "",
            field.constraints,
        )
        for name, field in fields.items()
    ]
    return _format_field_description_lines(entries=entries)
