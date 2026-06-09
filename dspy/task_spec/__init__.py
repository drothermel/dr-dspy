"""Public task_spec API.

Import spine symbols from this package. Internal modules (``wire``, ``type_registry``,
``invariants``, ``type_format``) are not part of the public surface.
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
from dspy.task_spec.invariants import validate_task_spec
from dspy.task_spec.parse import parse_task_spec_string
from dspy.task_spec.task_spec import TaskSpec
from dspy.task_spec.validation import validate_task_inputs, validate_task_inputs_from_spec

__all__ = [
    "FIELD_NAME_PATTERN",
    "FieldBinding",
    "FieldRole",
    "FieldSpec",
    "field_bindings",
    "field_desc_from_name",
    "TaskSpec",
    "default_task_instructions",
    "infer_prefix",
    "input_field",
    "make_task_spec",
    "output_field",
    "parse_task_spec_string",
    "validate_field_name",
    "validate_task_inputs",
    "validate_task_inputs_from_spec",
    "validate_task_spec",
]
