from __future__ import annotations

from dspy.adapters.types.tool import ToolCallResults
from dspy.task_spec import FieldSpec, TaskSpec, input_field


class ToolCallResultsTaskSpec(TaskSpec):
    name: str = "framework.adapter.tool_call_results"
    instructions: str = "Tool call results from conversation history."
    inputs: tuple[FieldSpec, ...] = (
        input_field(
            "tool_call_results",
            type_=ToolCallResults,
            desc="Serialized tool call results appended to the conversation history.",
        ),
    )
    outputs: tuple[FieldSpec, ...] = ()
