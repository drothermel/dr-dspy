from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from dspy.adapters.call.policies.parse_fallback import NoOpParseFallbackPolicy
from dspy.adapters.call.policies.response_format import NoOpResponseFormatPolicy
from dspy.adapters.call.two_step import TwoStepCallExecutor
from dspy.core.types.config import coerce_lm_config
from dspy.core.types.openai_compat import request_messages_as_openai
from dspy.errors import AdapterParseError, LMError
from dspy.predict.call_validation import validate_task_inputs
from dspy.runtime.transparency import resolve_call, resolve_call_site, resolve_lm_config, validate_compiled_call

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
            return await TwoStepCallExecutor.execute(
                cast("Any", adapter),
                lm=lm,
                config=coerce_lm_config(config),
                task_spec=task_spec,
                demos=demos,
                inputs=inputs,
                run=run,
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
            resolved_config = coerce_lm_config(config)
            original_field_names = set(task_spec.fields.keys())
            processed_task_spec, tools, resolved_config = adapter._call_preprocess(
                lm=lm, config=resolved_config, task_spec=task_spec, inputs=inputs
            )
            mutations = [
                f"removed field {name}"
                for name in sorted(original_field_names - set(processed_task_spec.fields.keys()))
            ]
            messages = adapter.format(task_spec=processed_task_spec, demos=demos, inputs=inputs)
            request = adapter._render_request(lm=lm, config=resolved_config, tools=tools, messages=messages)
            merged_config, provenance = resolve_lm_config(lm, resolved_config)
            site = resolve_call_site(
                run=run,
                call_site=call_site,
                default_module=type(adapter).__name__,
                default_phase="adapter",
            )
            compiled = resolve_call(
                lm=lm,
                adapter=adapter,
                task_spec=task_spec,
                processed_task_spec=processed_task_spec,
                config=merged_config,
                config_provenance=provenance,
                messages=request_messages_as_openai(request),
                task_spec_mutations=mutations,
                module=site.module,
                phase=site.phase,
                lm_role=site.lm_role,
            )
            transparency = run.telemetry.transparency
            validate_compiled_call(compiled, transparency)
            response = await adapter._call_lm(lm=lm, request=request, run=run, compiled=compiled)
            return adapter._call_postprocess(
                processed_task_spec=processed_task_spec,
                original_task_spec=task_spec,
                response=response,
                _lm=lm,
                _config=resolved_config,
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
