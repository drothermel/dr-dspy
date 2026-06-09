"""Public task_spec API.

Import spine symbols from this package. Internal modules (``type_registry``,
``validation`` helpers, ``annotation_format``) are not part of the public surface.
"""

from dspy.task_spec.bindings import FieldBinding, field_bindings
from dspy.task_spec.defaults import default_task_instructions
from dspy.task_spec.factory import make_task_spec
from dspy.task_spec.field_spec import (
    FIELD_NAME_PATTERN,
    FieldRole,
    FieldSpec,
    field_desc_from_name,
    infer_prefix,
    input_field,
    output_field,
    validate_field_name,
)
from dspy.task_spec.parse import parse_task_spec_string
from dspy.task_spec.serialize import TASK_SPEC_VERSION, field_spec_from_dict, field_spec_to_dict
from dspy.task_spec.task_spec import TaskSpec, validate_task_spec
from dspy.task_spec.validation import validate_task_inputs_from_spec

__all__ = [
    "FIELD_NAME_PATTERN",
    "FieldBinding",
    "FieldRole",
    "FieldSpec",
    "TASK_SPEC_VERSION",
    "field_bindings",
    "field_desc_from_name",
    "field_spec_from_dict",
    "field_spec_to_dict",
    "TaskSpec",
    "default_task_instructions",
    "infer_prefix",
    "input_field",
    "make_task_spec",
    "output_field",
    "parse_task_spec_string",
    "validate_field_name",
    "validate_task_inputs_from_spec",
    "validate_task_spec",
]
