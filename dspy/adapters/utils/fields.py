import enum
import inspect
import json
from collections.abc import Mapping
from typing import Any, Literal, cast, get_args, get_origin

import pydantic
from pydantic.fields import FieldInfo

from dspy.adapters.types.base_type import Type as DspyType
from dspy.adapters.types.code import Code
from dspy.adapters.types.reasoning import Reasoning
from dspy.adapters.utils.json import serialize_for_json
from dspy.task_spec.pydantic_bridge import get_dspy_field_type


def _annotation_is_subclass(annotation: object, expected_base: type) -> bool:
    try:
        return inspect.isclass(annotation) and issubclass(annotation, expected_base)
    except TypeError:
        return False


def format_field_value(field_info: FieldInfo, value: object, assume_text: bool = True) -> str | dict[str, str]:
    """
    Formats the value of the specified field according to the field's DSPy type (input or output),
    annotation (e.g. str, int, etc.), and the type of the value itself.

    Args:
      field_info: Information about the field, including its DSPy field type and annotation.
      value: The value of the field.
    Returns:
      The formatted value of the field, represented as a string.
    """
    string_value = None
    if isinstance(value, list) and field_info.annotation is str:
        # If the field has no special type requirements, format it as a nice numbered list for the LM.
        string_value = _format_input_list_field_value(value)
    else:
        jsonable_value = serialize_for_json(value)
        if isinstance(jsonable_value, (dict, list)):
            string_value = json.dumps(jsonable_value, ensure_ascii=False)
        else:
            # If the value is not a Python representation of a JSON object or Array
            # (e.g. the value is a JSON string), just use the string representation of the value
            # to avoid double-quoting the JSON string (which would hurt accuracy for certain
            # tasks, e.g. tasks that rely on computing string length)
            string_value = str(jsonable_value)

    if assume_text:
        return string_value
    return {"type": "text", "text": string_value}


def _get_json_schema(field_type: object) -> object:
    def move_type_to_front(d: object) -> object:
        # Move the 'type' key to the front of the dictionary, recursively, for LLM readability/adherence.
        if isinstance(d, Mapping):
            return {
                k: move_type_to_front(v) for k, v in sorted(d.items(), key=lambda item: (item[0] != "type", item[0]))
            }
        if isinstance(d, list):
            return [move_type_to_front(item) for item in d]
        return d

    schema = pydantic.TypeAdapter(field_type).json_schema()
    return move_type_to_front(schema)


def translate_field_type(field_name: str, field_info: FieldInfo) -> str:
    field_type = field_info.annotation

    if get_dspy_field_type(field_info) == "input" or field_type is str or field_type is Reasoning:
        desc = ""
    elif field_type is bool:
        desc = "must be True or False"
    elif field_type in (int, float):
        desc = f"must be a single {field_type.__name__} value"
    elif _annotation_is_subclass(annotation=field_type, expected_base=enum.Enum):
        enum_type = cast("type[enum.Enum]", field_type)
        enum_vals = "; ".join(str(member.value) for member in enum_type)
        desc = f"must be one of: {enum_vals}"
    elif hasattr(field_type, "__origin__") and field_type.__origin__ is Literal:
        desc = (
            # Strongly encourage the LM to avoid choosing values that don't appear in the
            # literal or returning a value of the form 'Literal[<selected_value>]'
            f"must exactly match (no extra characters) one of: {'; '.join([str(x) for x in field_type.__args__])}"
        )
    elif (
        _annotation_is_subclass(annotation=field_type, expected_base=Code)
        and cast("type[Code]", field_type).description()
    ):
        # Code has a rich type description already; avoid duplicating its large schema block.
        desc = ""
    else:
        desc = f"must adhere to the JSON schema: {json.dumps(_get_json_schema(field_type), ensure_ascii=False)}"

    desc = (" " * 8) + f"# note: the value you produce {desc}" if desc else ""
    return f"{{{field_name}}}{desc}"


def get_annotation_name(annotation: object) -> str:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is None:
        if hasattr(annotation, "__name__"):
            return cast("str", annotation.__name__)
        return str(annotation)

    if origin is Literal:
        args_str = ", ".join(
            _quoted_string_for_literal_type_annotation(a) if isinstance(a, str) else get_annotation_name(a)
            for a in args
        )
        return f"{get_annotation_name(origin)}[{args_str}]"
    args_str = ", ".join(get_annotation_name(a) for a in args)
    return f"{get_annotation_name(origin)}[{args_str}]"


def get_field_description_string(fields: dict[str, FieldInfo]) -> str:
    field_descriptions = []
    for idx, (k, v) in enumerate(fields.items()):
        extra = cast("dict[str, Any]", v.json_schema_extra or {})
        field_message = f"{idx + 1}. `{k}`"
        field_message += f" ({get_annotation_name(v.annotation)})"
        desc = extra["desc"] if extra.get("desc") != f"${{{k}}}" else ""

        custom_types = DspyType.extract_custom_type_from_annotation(v.annotation)
        for custom_type in custom_types:
            if len(custom_type.description()) > 0:
                desc += f"\n    Type description of {get_annotation_name(custom_type)}: {custom_type.description()}"

        field_message += f": {desc}"
        field_message += f"\nConstraints: {extra['constraints']}" if extra.get("constraints") else ""
        field_descriptions.append(field_message)
    return "\n".join(field_descriptions).strip()


def _format_input_list_field_value(value: list[Any]) -> str:
    """
    Formats the value of an input field of type list[Any].

    Args:
      value: The value of the list-type input field.
    Returns:
      A string representation of the input field's list value.
    """
    if len(value) == 0:
        return "N/A"
    if len(value) == 1:
        return _format_blob(str(value[0]))

    return "\n".join([f"[{idx + 1}] {_format_blob(str(txt))}" for idx, txt in enumerate(value)])


def _format_blob(blob: str) -> str:
    """
    Formats the specified text blobs so that an LM can parse it correctly within a list
    of multiple text blobs.

    Args:
        blob: The text blob to format.
    Returns:
        The formatted text blob.
    """
    if "\n" not in blob and "«" not in blob and "»" not in blob:
        return f"«{blob}»"

    modified_blob = blob.replace("\n", "\n    ")
    return f"«««\n    {modified_blob}\n»»»"


def _quoted_string_for_literal_type_annotation(s: str) -> str:
    """
    Return the specified string quoted for inclusion in a literal type annotation.
    """
    has_single = "'" in s
    has_double = '"' in s

    if has_single and not has_double:
        # Only single quotes => enclose in double quotes
        return f'"{s}"'
    if has_double and not has_single:
        # Only double quotes => enclose in single quotes
        return f"'{s}'"
    if has_single and has_double:
        # Both => enclose in single quotes; escape each single quote with \'
        escaped = s.replace("'", "\\'")
        return f"'{escaped}'"
    # Neither => enclose in single quotes
    return f"'{s}'"
