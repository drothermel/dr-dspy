"""Field description formatting for TaskSpec instances."""

from dspy.adapters.types.base_type import Type as DspyType
from dspy.adapters.utils import get_annotation_name
from dspy.task_spec.field_spec import FieldSpec


def get_field_spec_description_string(fields: dict[str, FieldSpec]) -> str:
    """Format input or output fields for adapter system prompts."""
    field_descriptions = []
    for idx, (name, field) in enumerate(fields.items()):
        field_message = f"{idx + 1}. `{name}`"
        field_message += f" ({get_annotation_name(field.type_)})"
        desc = field.desc if field.desc != f"${{{name}}}" else ""

        custom_types = DspyType.extract_custom_type_from_annotation(field.type_)
        for custom_type in custom_types:
            if len(custom_type.description()) > 0:
                desc += f"\n    Type description of {get_annotation_name(custom_type)}: {custom_type.description()}"

        field_message += f": {desc}"
        if field.constraints:
            field_message += f"\nConstraints: {field.constraints}"
        field_descriptions.append(field_message)
    return "\n".join(field_descriptions).strip()
