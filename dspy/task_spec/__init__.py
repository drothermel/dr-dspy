from dspy.task_spec.defaults import default_task_instructions
from dspy.task_spec.factory import make_task_spec
from dspy.task_spec.field_spec import FieldSpec, infer_prefix
from dspy.task_spec.task_spec import TaskSpec

__all__ = [
    "FieldSpec",
    "TaskSpec",
    "default_task_instructions",
    "infer_prefix",
    "make_task_spec",
]
