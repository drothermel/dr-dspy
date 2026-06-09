from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dspy.adapters.call.tool_output import attach_tool_calls_to_value
from dspy.core.types import LMRequest
from dspy.core.types.config import coerce_lm_config, merge_lm_request_config
from dspy.core.types.openai_compat import request_messages_as_openai
from dspy.transparency.resolve import resolve_call, resolve_lm_config
from dspy.utils.exceptions import AdapterParseError, LMError
from dspy.utils.transparency import validate_compiled_call

if TYPE_CHECKING:
    from dspy.adapters.two_step_adapter import TwoStepAdapter
    from dspy.clients.base_lm import BaseLM
    from dspy.core.types.config import LMConfig
    from dspy.runtime.run_context import RunContext
    from dspy.task_spec import TaskSpec


class TwoStepCallExecutor:
    @staticmethod
    async def execute(
        adapter: TwoStepAdapter,
        *,
        lm: BaseLM,
        config: LMConfig | None,
        task_spec: TaskSpec,
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
        run: RunContext,
    ) -> list[dict[str, Any]]:
        resolved_config = coerce_lm_config(config)
        messages = adapter.format(task_spec=task_spec, demos=demos, inputs=inputs)
        merged_config, provenance = resolve_lm_config(lm, resolved_config)
        request = LMRequest(
            model=lm.model,
            messages=messages,
            config=merge_lm_request_config(lm=lm, config=merged_config),
        )
        transparency = run.telemetry.transparency
        main_compiled = resolve_call(
            lm=lm,
            adapter=adapter,
            task_spec=task_spec,
            config=merged_config,
            config_provenance=provenance,
            messages=request_messages_as_openai(request),
            module="TwoStepAdapter",
            phase="two_step.main",
            lm_role="default",
        )
        validate_compiled_call(main_compiled, transparency)
        response = await lm(request, run=run, compiled=main_compiled)

        extractor_task_spec = adapter._create_extractor_task_spec(task_spec)
        values = []
        for output in response.outputs:
            output_logprobs = output.logprobs
            text = output.text
            try:
                value = await adapter._run_extraction(original_task_spec=task_spec, text=text or "", run=run)
            except LMError:
                raise
            except Exception as e:
                raise AdapterParseError(
                    adapter_name="TwoStepAdapter",
                    task_spec=extractor_task_spec,
                    lm_response=str(output),
                    message=f"Failed to parse response from the original completion: {e}",
                ) from e

            value = attach_tool_calls_to_value(
                value=value,
                output=output,
                original_task_spec=task_spec,
                get_tool_call_output_field_name=adapter._get_tool_call_output_field_name,
            )
            if output_logprobs is not None:
                value["logprobs"] = output_logprobs
            values.append(value)
        return values
