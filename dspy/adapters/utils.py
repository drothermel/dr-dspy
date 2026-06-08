import ast
import enum
import inspect
import json
import types
from collections.abc import Mapping
from typing import Any, Literal, Union, cast, get_args, get_origin

import json_repair
import pydantic
from pydantic import TypeAdapter
from pydantic.fields import FieldInfo

from dspy.adapters.types.base_type import Type as DspyType
from dspy.adapters.types.code import Code
from dspy.adapters.types.reasoning import Reasoning
from dspy.core.types import LMMessage
from dspy.signatures.signature import Signature
from dspy.signatures.utils import get_dspy_field_type


def build_lm_message(
    role: str,
    content: str | list[dict[str, Any]] | None = None,
    **extra: Any,
) -> LMMessage:
    payload: dict[str, Any] = {"role": role}
    if content is not None:
        payload["content"] = content
    payload.update(extra)
    return LMMessage(**payload)


def _annotation_is_subclass(annotation: object, expected_base: type) -> bool:
    try:
        return inspect.isclass(annotation) and issubclass(annotation, expected_base)
    except TypeError:
        return False


def serialize_for_json(value: object) -> object:
    """
    Formats the specified value so that it can be serialized as a JSON string.

    Args:
        value: The value to format as a JSON string.
    Returns:
        The formatted value, which is serializable as a JSON string.
    """
    # Attempt to format the value as a JSON-compatible object using pydantic, falling back to
    # a string representation of the value if that fails (e.g. if the value contains an object
    # that pydantic doesn't recognize or can't serialize)
    try:
        return TypeAdapter(type(value)).dump_python(value, mode="json")
    except Exception:
        return str(value)


def _parse_serialized_content_block_string(value: object) -> list[dict[str, Any]] | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped.startswith("["):
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list) or not parsed:
        return None
    if all(isinstance(block, dict) and "type" in block for block in parsed):
        return cast("list[dict[str, Any]]", parsed)
    return None


def value_contains_multimodal_custom_type(value: object) -> bool:
    if isinstance(value, DspyType):
        return value.renders_as_content_blocks()
    if _parse_serialized_content_block_string(value) is not None:
        return True
    if isinstance(value, list):
        return any(value_contains_multimodal_custom_type(item) for item in value)
    if isinstance(value, dict):
        return any(value_contains_multimodal_custom_type(item) for item in value.values())
    if isinstance(value, pydantic.BaseModel):
        return any(value_contains_multimodal_custom_type(getattr(value, name)) for name in type(value).model_fields)
    return False


def inputs_include_multimodal_custom_type_values(signature: type[Signature], inputs: Mapping[str, Any]) -> bool:
    for field_name in signature.input_fields:
        if field_name in inputs and value_contains_multimodal_custom_type(inputs[field_name]):
            return True
    return False


def collect_multimodal_content_blocks(value: object) -> list[dict[str, Any]]:
    if isinstance(value, DspyType):
        return value.to_content_blocks() if value.renders_as_content_blocks() else []
    if blocks := _parse_serialized_content_block_string(value):
        return blocks
    if isinstance(value, list):
        blocks: list[dict[str, Any]] = []
        for item in value:
            blocks.extend(collect_multimodal_content_blocks(item))
        return blocks
    if isinstance(value, dict):
        blocks = []
        for item in value.values():
            blocks.extend(collect_multimodal_content_blocks(item))
        return blocks
    if isinstance(value, pydantic.BaseModel):
        blocks = []
        for name in type(value).model_fields:
            blocks.extend(collect_multimodal_content_blocks(getattr(value, name)))
        return blocks
    return []


def field_value_to_content_blocks(
    field_info: FieldInfo,
    field_name: str,
    value: object,
    *,
    prefix: str = "",
    field_wrapper: str | None = None,
) -> list[dict[str, Any]]:
    if field_wrapper == "xml":
        open_tag = f"{prefix}<{field_name}>\n"
        close_tag = f"\n</{field_name}>"
        if isinstance(value, DspyType) and value.renders_as_content_blocks():
            return [{"type": "text", "text": open_tag}, *value.to_content_blocks(), {"type": "text", "text": close_tag}]
        nested_blocks = collect_multimodal_content_blocks(value)
        if nested_blocks:
            return [{"type": "text", "text": open_tag}, *nested_blocks, {"type": "text", "text": close_tag}]
        formatted_field_value = format_field_value(field_info=field_info, value=value)
        return [{"type": "text", "text": f"{open_tag}{formatted_field_value}{close_tag}"}]

    header = f"{prefix}[[ ## {field_name} ## ]]\n"
    if isinstance(value, DspyType) and value.renders_as_content_blocks():
        return [{"type": "text", "text": header}, *value.to_content_blocks()]
    nested_blocks = collect_multimodal_content_blocks(value)
    if nested_blocks:
        return [{"type": "text", "text": header}, *nested_blocks]
    formatted_field_value = format_field_value(field_info=field_info, value=value)
    return [{"type": "text", "text": f"{header}{formatted_field_value}"}]


def build_multimodal_user_message_content(
    signature: type[Signature],
    inputs: Mapping[str, Any],
    *,
    prefix: str = "",
    suffix: str = "",
    main_request: bool = False,
    output_requirements: str | None = None,
    field_wrapper: str | None = None,
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if prefix:
        blocks.append({"type": "text", "text": prefix})

    field_blocks_added = False
    for field_name, field_info in signature.input_fields.items():
        if field_name not in inputs:
            continue
        field_prefix = "\n\n" if field_blocks_added else ""
        field_blocks_added = True
        blocks.extend(
            field_value_to_content_blocks(
                field_info,
                field_name,
                inputs[field_name],
                prefix=field_prefix,
                field_wrapper=field_wrapper,
            )
        )

    if main_request and output_requirements is not None:
        blocks.append({"type": "text", "text": f"\n\n{output_requirements}"})

    if suffix:
        blocks.append({"type": "text", "text": suffix})
    return blocks


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
    elif _annotation_is_subclass(field_type, enum.Enum):
        enum_type = cast("type[enum.Enum]", field_type)
        enum_vals = "; ".join(str(member.value) for member in enum_type)
        desc = f"must be one of: {enum_vals}"
    elif hasattr(field_type, "__origin__") and field_type.__origin__ is Literal:
        desc = (
            # Strongly encourage the LM to avoid choosing values that don't appear in the
            # literal or returning a value of the form 'Literal[<selected_value>]'
            f"must exactly match (no extra characters) one of: {'; '.join([str(x) for x in field_type.__args__])}"
        )
    elif _annotation_is_subclass(field_type, Code) and cast("type[Code]", field_type).description():
        # Code has a rich type description already; avoid duplicating its large schema block.
        desc = ""
    else:
        desc = f"must adhere to the JSON schema: {json.dumps(_get_json_schema(field_type), ensure_ascii=False)}"

    desc = (" " * 8) + f"# note: the value you produce {desc}" if desc else ""
    return f"{{{field_name}}}{desc}"


def find_enum_member(enum_type: enum.EnumMeta, identifier: object) -> enum.Enum:
    """
    Finds the enum member corresponding to the specified identifier, which may be the
    enum member's name or value.

    Args:
        enum: The enum to search for the member.
        identifier: If the enum is explicitly-valued, this is the value of the enum member to find.
                    If the enum is auto-valued, this is the name of the enum member to find.
    Returns:
        The enum member corresponding to the specified identifier.
    """
    # Check if the identifier is a valid enum member value *before* checking if it's a valid enum
    # member name, since the identifier will be a value for explicitly-valued enums. This handles
    # the (rare) case where an enum member value is the same as another enum member's name in
    # an explicitly-valued enum
    for member in enum_type:
        member = cast("enum.Enum", member)
        if member.value == identifier:
            return member

    # If the identifier is not a valid enum member value, check if it's a valid enum member name,
    # since the identifier will be a member name for auto-valued enums
    if isinstance(identifier, str) and identifier in enum_type.__members__:
        return cast("enum.Enum", enum_type[identifier])

    raise ValueError(f"{identifier} is not a valid name or value for the enum {enum_type.__name__}")


def parse_value(value: object, annotation: object) -> object:
    if annotation is str:
        return str(value)

    if isinstance(annotation, enum.EnumMeta):
        return find_enum_member(annotation, value)

    origin = get_origin(annotation)

    if origin is Literal:
        allowed = get_args(annotation)
        if value in allowed:
            return value

        if isinstance(value, str):
            v = value.strip()
            if v.startswith(("Literal[", "str[")) and v.endswith("]"):
                v = v[v.find("[") + 1 : -1]
            if len(v) > 1 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1]

            if v in allowed:
                return v

        raise ValueError(f"{value!r} is not one of {allowed!r}")

    if not isinstance(value, str):
        return TypeAdapter(annotation).validate_python(value)

    if origin in (Union, types.UnionType) and type(None) in get_args(annotation) and str in get_args(annotation):
        # Handle union annotations such as `str | None`.
        return TypeAdapter(annotation).validate_python(value)

    candidate = json_repair.loads(value)  # json_repair.loads returns "" on failure.
    if candidate == "" and value != "":
        try:
            candidate = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            candidate = value

    try:
        return TypeAdapter(annotation).validate_python(candidate)
    except pydantic.ValidationError as e:
        if _annotation_is_subclass(annotation, DspyType):
            try:
                # For dspy.Type, try parsing from the original value in case it has a custom parser
                return TypeAdapter(annotation).validate_python(value)
            except Exception:
                raise e
        raise


def get_annotation_name(annotation: object) -> str:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is None:
        if annotation is Reasoning:
            # Keep backward compatibility with the old behavior in `ChainOfThought`, where reasoning
            # field type is treated as a string.
            return "str"
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
