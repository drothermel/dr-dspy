"""Recursive Language Model (RLM) predictor."""

from dspy.predict.rlm.module import RLM
from dspy.predict.rlm.sync_bridge import _strip_code_fences
from dspy.predict.rlm.task_specs import FrameworkRlmSubQueryTaskSpec

__all__ = ["RLM", "FrameworkRlmSubQueryTaskSpec", "_strip_code_fences"]
