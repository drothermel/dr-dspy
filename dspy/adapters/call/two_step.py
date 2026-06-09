from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dspy.adapters.base.tool_calls import attach_tool_calls_to_parsed_value
from dspy.runtime.config import CallSite

if TYPE_CHECKING:
    from dspy.adapters.call.two_step_protocol import TwoStepMainAdapter
    from dspy.core.types import LMResponse
    from dspy.runtime.run_context import RunContext
    from dspy.task_spec import TaskSpec


async def finalize_two_step_main_response(
    adapter: TwoStepMainAdapter,
    *,
    response: LMResponse,
    original_task_spec: TaskSpec,
    run: RunContext,
) -> list[dict[str, Any]]:
    values = []
    for output in response.outputs:
        output_logprobs = output.logprobs
        text = output.text
        value = await adapter._run_extraction(original_task_spec=original_task_spec, text=text or "", run=run)
        value = attach_tool_calls_to_parsed_value(
            value=value,
            output=output,
            tool_call_output_field_name=adapter._get_tool_call_output_field_name(original_task_spec),
        )
        if output_logprobs is not None:
            value["logprobs"] = output_logprobs
        values.append(value)
    return values


TWO_STEP_MAIN_CALL_SITE = CallSite(module="TwoStepAdapter", phase="two_step.main", lm_role="default")
