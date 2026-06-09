from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from dspy.adapters.call.policies.parse_fallback import NoOpParseFallbackPolicy
from dspy.adapters.call.policies.response_format import NoOpResponseFormatPolicy
from dspy.adapters.call.stages import invoke_adapter_lm, prepare_adapter_call
from dspy.adapters.call.two_step import TWO_STEP_MAIN_CALL_SITE, finalize_two_step_main_response
from dspy.core.types.config import coerce_lm_config
from dspy.errors import AdapterParseError, LMError
from dspy.predict.call_validation import validate_task_inputs

if TYPE_CHECKING:
    from collections.abc import Mapping

    from dspy.adapters.base.adapter import Adapter
    from dspy.clients.base_lm import BaseLM
    from dspy.core.types.config import LMConfig
    from dspy.runtime.config import CallSite
    from dspy.runtime.run_context import RunContext
    from dspy.task_spec import TaskSpec


class AdapterCallPipeline:
    @staticmethod
    async def execute(
        adapter: Adapter,
        *,
        lm: BaseLM,
        config: LMConfig | Mapping[str, Any] | None,
        task_spec: TaskSpec,
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
        run: RunContext,
        call_site: CallSite | None = None,
        allow_parse_fallback: bool = True,
    ) -> list[dict[str, Any]]:
        inputs = validate_task_inputs(task_spec, inputs)
        if getattr(adapter, "call_mode", None) == "two_step":
            return await AdapterCallPipeline._run_two_step_call(
                adapter,
                lm=lm,
                config=config,
                task_spec=task_spec,
                demos=demos,
                inputs=inputs,
                run=run,
                call_site=call_site,
            )

        response_format_policy = adapter.response_format_policy or NoOpResponseFormatPolicy()

        async def run_once(effective_config: LMConfig | None) -> list[dict[str, Any]]:
            return await AdapterCallPipeline._run_single_call(
                adapter,
                lm=lm,
                config=effective_config,
                task_spec=task_spec,
                demos=demos,
                inputs=inputs,
                run=run,
                call_site=call_site,
                allow_parse_fallback=allow_parse_fallback,
            )

        return await response_format_policy.execute(
            adapter=adapter,
            lm=lm,
            config=coerce_lm_config(config),
            task_spec=task_spec,
            demos=demos,
            inputs=inputs,
            run_once=run_once,
        )

    @staticmethod
    async def _run_two_step_call(
        adapter: Adapter,
        *,
        lm: BaseLM,
        config: LMConfig | Mapping[str, Any] | None,
        task_spec: TaskSpec,
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
        run: RunContext,
        call_site: CallSite | None,
    ) -> list[dict[str, Any]]:
        response_format_policy = adapter.response_format_policy or NoOpResponseFormatPolicy()

        async def run_once(effective_config: LMConfig | None) -> list[dict[str, Any]]:
            prepared = prepare_adapter_call(
                adapter,
                lm=lm,
                config=effective_config,
                task_spec=task_spec,
                demos=demos,
                inputs=inputs,
            )
            response = await invoke_adapter_lm(
                adapter,
                prepared,
                lm=lm,
                run=run,
                call_site=call_site or TWO_STEP_MAIN_CALL_SITE,
                default_module="TwoStepAdapter",
                default_phase="two_step.main",
            )
            return await finalize_two_step_main_response(
                cast("Any", adapter),
                response=response,
                original_task_spec=task_spec,
                run=run,
            )

        return await response_format_policy.execute(
            adapter=adapter,
            lm=lm,
            config=coerce_lm_config(config),
            task_spec=task_spec,
            demos=demos,
            inputs=inputs,
            run_once=run_once,
        )

    @staticmethod
    async def _run_single_call(
        adapter: Adapter,
        *,
        lm: BaseLM,
        config: LMConfig | Mapping[str, Any] | None,
        task_spec: TaskSpec,
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
        run: RunContext,
        call_site: CallSite | None,
        allow_parse_fallback: bool,
    ) -> list[dict[str, Any]]:
        try:
            prepared = prepare_adapter_call(
                adapter,
                lm=lm,
                config=config,
                task_spec=task_spec,
                demos=demos,
                inputs=inputs,
            )
            response = await invoke_adapter_lm(
                adapter,
                prepared,
                lm=lm,
                run=run,
                call_site=call_site,
            )
            return adapter._call_postprocess(
                processed_task_spec=prepared.processed_task_spec,
                original_task_spec=task_spec,
                response=response,
                _lm=lm,
                _config=prepared.config,
            )
        except TypeError:
            raise
        except LMError:
            raise
        except AdapterParseError as error:
            if not allow_parse_fallback:
                raise
            parse_fallback_policy = adapter.parse_fallback_policy or NoOpParseFallbackPolicy()
            return await parse_fallback_policy.execute_fallback(
                adapter=adapter,
                lm=lm,
                config=coerce_lm_config(config),
                task_spec=task_spec,
                demos=demos,
                inputs=inputs,
                run=run,
                error=error,
            )
