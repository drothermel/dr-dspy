"""
Custom adapter for improving structured outputs using the information from Pydantic models.
Based on the format used by BAML: https://github.com/BoundaryML/baml
"""

import inspect
import types
from typing import Any, Literal, Union, cast, get_args, get_origin

from pydantic import BaseModel
from typing_extensions import override

from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.utils import build_multimodal_user_message_content, inputs_include_multimodal_custom_type_values
from dspy.adapters.utils import format_field_value as original_format_field_value
from dspy.signatures.signature import Signature

# BAML schema comments are prompt text; this adapter uses Python-style # comments because they have produced better model adherence than //.
COMMENT_SYMBOL = "#"
INDENTATION = "  "


def _render_type_str(
    annotation: object,
    depth: int = 0,
    indent: int = 0,
    seen_models: set[type] | None = None,
) -> str:
    """Recursively renders a type annotation into a simplified string.

    Args:
        annotation: The type annotation to render
        depth: Current recursion depth (prevents infinite recursion)
        indent: Current indentation level for nested structures
    """
    if annotation is str:
        return "string"
    if annotation is int:
        return "int"
    if annotation is float:
        return "float"
    if annotation is bool:
        return "boolean"
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        return _build_simplified_schema(pydantic_model=annotation, indent=indent, seen_models=seen_models)

    try:
        origin = get_origin(annotation)
        args = get_args(annotation)
    except Exception:
        return str(annotation)

    if origin in (types.UnionType, Union):
        non_none_args = [arg for arg in args if arg is not type(None)]
        type_render = " or ".join(
            [
                _render_type_str(annotation=arg, depth=depth + 1, indent=indent, seen_models=seen_models)
                for arg in non_none_args
            ]
        )
        if len(non_none_args) < len(args):
            return f"{type_render} or null"
        return type_render

    if origin is Literal:
        return " or ".join(f'"{arg}"' for arg in args)

    if origin is list:
        inner_type = args[0]
        if inspect.isclass(inner_type) and issubclass(inner_type, BaseModel):
            inner_schema = _build_simplified_schema(
                pydantic_model=inner_type, indent=indent + 1, seen_models=seen_models
            )
            current_indent = INDENTATION * indent
            return f"[\n{inner_schema}\n{current_indent}]"
        return f"{_render_type_str(annotation=inner_type, depth=depth + 1, indent=indent, seen_models=seen_models)}[]"

    if origin is dict:
        return f"dict[{_render_type_str(annotation=args[0], depth=depth + 1, indent=indent, seen_models=seen_models)}, {_render_type_str(annotation=args[1], depth=depth + 1, indent=indent, seen_models=seen_models)}]"

    if hasattr(annotation, "__name__"):
        return cast("str", annotation.__name__)
    return str(annotation)


def _build_simplified_schema(
    pydantic_model: type[BaseModel],
    indent: int = 0,
    seen_models: set[type] | None = None,
) -> str:
    """Builds a simplified, human-readable schema from a Pydantic model.

    Args:
        pydantic_model: The Pydantic model to build schema for
        indent: Current indentation level
        seen_models: Set to track visited pydantic models (prevents infinite recursion)
    """
    seen_models = seen_models or set()

    if pydantic_model in seen_models:
        raise ValueError("BAMLAdapter cannot handle recursive pydantic models, please use a different adapter.")

    # Add `pydantic_model` to `seen_models` with a placeholder value to avoid infinite recursion.
    seen_models.add(pydantic_model)

    lines = []
    current_indent = INDENTATION * indent
    next_indent = INDENTATION * (indent + 1)

    if indent == 0 and pydantic_model.__doc__:
        docstring = pydantic_model.__doc__.strip()
        # Handle multiline docstrings by prefixing each line with the comment symbol
        for line in docstring.split("\n"):
            line = line.strip()
            if line:
                lines.append(f"{current_indent}{COMMENT_SYMBOL} {line}")

    lines.append(f"{current_indent}{{")

    fields = pydantic_model.model_fields
    if not fields:
        lines.append(f"{next_indent}{COMMENT_SYMBOL} No fields defined")
    for name, field in fields.items():
        if field.description:
            lines.append(f"{next_indent}{COMMENT_SYMBOL} {field.description}")
        elif field.alias and field.alias != name:
            # If there's an alias but no description, show the alias as a comment
            lines.append(f"{next_indent}{COMMENT_SYMBOL} alias: {field.alias}")

        # If the field type is a BaseModel, add its docstring as a comment before the field
        field_annotation = field.annotation
        # Handle union types
        origin = get_origin(field_annotation)
        if origin in (types.UnionType, Union):
            args = get_args(field_annotation)
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1:
                field_annotation = non_none_args[0]

        if inspect.isclass(field_annotation) and issubclass(field_annotation, BaseModel) and field_annotation.__doc__:
            docstring = field_annotation.__doc__.strip()
            for line in docstring.split("\n"):
                line = line.strip()
                if line:
                    lines.append(f"{next_indent}{COMMENT_SYMBOL} {line}")

        rendered_type = _render_type_str(annotation=field.annotation, indent=indent + 1, seen_models=seen_models)
        line = f"{next_indent}{name}: {rendered_type},"

        lines.append(line)

    lines.append(f"{current_indent}}}")
    return "\n".join(lines)


class BAMLAdapter(JSONAdapter):
    """
    A DSPy adapter that improves the rendering of complex/nested Pydantic models to help LMs.

    This adapter generates a compact, human-readable schema representation for nested Pydantic output
    fields, inspired by the BAML project's JSON formatter (https://github.com/BoundaryML/baml).
    The resulting rendered schema is more token-efficient and easier for smaller LMs to follow than a
    raw JSON schema. It also includes Pydantic field descriptions as comments in the schema, which
    provide valuable additional context for the LM to understand the expected output.

    Example Usage:
    ```python
    from typing import Literal

    from pydantic import BaseModel, Field

    from dspy.adapters.baml_adapter import BAMLAdapter
    from dspy.clients.lm import LM
    from dspy.dsp.utils.settings import settings
    from dspy.predict.predict import Predict
    from dspy.signatures.field import InputField, OutputField
    from dspy.signatures.signature import Signature

    # 1. Define your Pydantic models
    class PatientAddress(BaseModel):
        street: str
        city: str
        country: Literal["US", "CA"]

    class PatientDetails(BaseModel):
        name: str = Field(description="Full name of the patient.")
        age: int
        address: PatientAddress | None

    # 2. Define a signature using the Pydantic model as an output field
    class ExtractPatientInfo(Signature):
        '''Extract patient information from the clinical note.'''
        clinical_note: str = InputField()
        patient_info: PatientDetails = OutputField()

    # 3. Configure DSPy to use the new adapter
    lm = LM("openai/gpt-4.1-mini")
    settings.configure(lm=lm, adapter=BAMLAdapter())

    # 4. Run your program
    extractor = Predict(ExtractPatientInfo)
    note = "John Doe, 45 years old, lives at 123 Main St, Anytown. Resident of the US."
    result = extractor(clinical_note=note)
    print(result.patient_info)

    # Expected output:
    # PatientDetails(name='John Doe', age=45, address=PatientAddress(street='123 Main St', city='Anytown', country='US'))
    ```
    """

    @override
    def format_field_structure(self, signature: type[Signature]) -> str:
        """Overrides the base method to generate a simplified schema for Pydantic models."""

        sections = []

        sections.append(
            "All interactions will be structured in the following way, with the appropriate values filled in.\n"
        )

        if signature.input_fields:
            for name in signature.input_fields:
                sections.append(f"[[ ## {name} ## ]]")
                sections.append(f"{{{name}}}")
                sections.append("")  # Empty line after each input

        if signature.output_fields:
            for name, field in signature.output_fields.items():
                field_type = field.annotation
                sections.append(f"[[ ## {name} ## ]]")
                sections.append(
                    f"Output field `{name}` should be of type: {_render_type_str(annotation=field_type, indent=0)}\n"
                )

        sections.append("[[ ## completed ## ]]")

        return "\n".join(sections)

    @override
    def format_user_message_content(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str | list[dict[str, Any]]:
        """Overrides the base method to render Pydantic input instances as clean JSON."""
        if inputs_include_multimodal_custom_type_values(signature=signature, inputs=inputs):
            output_requirements = self.user_message_output_requirements(signature) if main_request else None
            return build_multimodal_user_message_content(
                signature=signature,
                inputs=inputs,
                prefix=prefix,
                suffix=suffix,
                main_request=main_request,
                output_requirements=output_requirements,
            )

        messages = [prefix]
        for key, field_info in signature.input_fields.items():
            if key in inputs:
                value = inputs.get(key)
                formatted_value = ""
                if isinstance(value, BaseModel):
                    formatted_value = value.model_dump_json(indent=2, by_alias=True)
                else:
                    formatted_value = original_format_field_value(field_info=field_info, value=value)

                messages.append(f"[[ ## {key} ## ]]\n{formatted_value}")

        if main_request:
            output_requirements = self.user_message_output_requirements(signature)
            if output_requirements is not None:
                messages.append(output_requirements)

        messages.append(suffix)
        return "\n\n".join(m for m in messages if m).strip()
