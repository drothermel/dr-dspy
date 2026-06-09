from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from dspy.adapters.base.native import AdapterNativeMixin
from dspy.adapters.base.protocols import ComposedAdapterT
from dspy.adapters.call.postprocess import enrich_parsed_value_from_lm_output
from dspy.core.types import (
    LMConfig,
    LMMessage,
    LMRequest,
    LMResponse,
    LMToolSpec,
    merge_lm_request_config,
)
from dspy.errors import AdapterParseError
from dspy.runtime.run_context import RunContext
from dspy.task_spec import TaskSpec

if TYPE_CHECKING:
    from dspy.adapters.base.adapter import Adapter
    from dspy.clients.base_lm import BaseLM
    from dspy.runtime.config import CallSite


class AdapterCallMixin(AdapterNativeMixin):
    def _call_preprocess(
        self: ComposedAdapterT,
        lm: BaseLM,
        config: LMConfig | Mapping[str, Any],
        task_spec: TaskSpec,
        inputs: dict[str, Any],
    ) -> tuple[TaskSpec, list[LMToolSpec], LMConfig]:
        return self.preprocessor_chain.run(
            self,
            lm=lm,
            config=config,
            task_spec=task_spec,
            inputs=inputs,
        )

    def _call_postprocess(
        self: ComposedAdapterT,
        processed_task_spec: TaskSpec,
        original_task_spec: TaskSpec,
        response: LMResponse,
    ) -> list[dict[str, Any]]:
        values = []
        tool_call_output_field_name = self._get_tool_call_output_field_name(original_task_spec)
        for output in response.outputs:
            tool_calls = output.tool_calls
            text = output.text
            if text is not None and (not (tool_calls and tool_call_output_field_name)):
                value = self.parse(task_spec=processed_task_spec, completion=text)
            elif tool_calls and tool_call_output_field_name:
                value = (
                    self.parse(task_spec=processed_task_spec, completion=text)
                    if text and processed_task_spec.output_fields
                    else {}
                )
            elif text is None and (not processed_task_spec.output_fields):
                value = {}
            else:
                raise AdapterParseError(
                    adapter_name=type(self).__name__,
                    task_spec=original_task_spec,
                    lm_response=str(output),
                    message="The LM returned an empty or null response.",
                )
            value = enrich_parsed_value_from_lm_output(
                self,
                value=value,
                output=output,
                original_task_spec=original_task_spec,
            )
            values.append(value)
        return values

    def _render_request(
        self, lm: BaseLM, config: LMConfig, tools: list[LMToolSpec], messages: Sequence[LMMessage]
    ) -> LMRequest:
        return LMRequest(
            model=lm.model, messages=list(messages), tools=tools, config=merge_lm_request_config(lm=lm, config=config)
        )

    async def _call_lm(self, lm: BaseLM, request: LMRequest, *, run: RunContext, compiled=None) -> LMResponse:
        return await lm(request, run=run, compiled=compiled)

    async def __call__(
        self,
        *,
        lm: BaseLM,
        config: LMConfig | Mapping[str, Any] | None,
        task_spec: TaskSpec,
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
        run: RunContext,
        call_site: CallSite | None = None,
    ) -> list[dict[str, Any]]:
        from dspy.adapters.call.pipeline import AdapterCallPipeline

        return await AdapterCallPipeline.execute(
            cast("Adapter", self),
            lm=lm,
            config=config,
            task_spec=task_spec,
            demos=demos,
            inputs=inputs,
            run=run,
            call_site=call_site,
        )
