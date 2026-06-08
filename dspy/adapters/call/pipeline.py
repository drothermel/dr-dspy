from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from dspy.adapters.call.policies.parse_fallback import NoOpParseFallbackPolicy
from dspy.adapters.call.policies.response_format import NoOpResponseFormatPolicy
from dspy.core.types.config import coerce_lm_config
from dspy.utils.exceptions import AdapterParseError, LMError

if TYPE_CHECKING:
    from dspy.adapters.base.adapter import Adapter
    from dspy.clients.base_lm import BaseLM
    from dspy.core.types.config import LMConfig
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
        allow_parse_fallback: bool = True,
    ) -> list[dict[str, Any]]:
        if getattr(adapter, "call_mode", None) == "two_step":
            from dspy.adapters.call.two_step import TwoStepCallExecutor

            return await TwoStepCallExecutor.execute(
                cast("Any", adapter),
                lm=lm,
                config=coerce_lm_config(config),
                task_spec=task_spec,
                demos=demos,
                inputs=inputs,
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
                allow_parse_fallback=allow_parse_fallback,
            )

        return await response_format_policy.execute(
            adapter=adapter,
            lm=lm,
            config=config,
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
        allow_parse_fallback: bool,
    ) -> list[dict[str, Any]]:
        from dspy.compile.resolve import resolve_call, resolve_lm_config
        from dspy.core.types.history import _history_request_messages_as_openai
        from dspy.dsp.utils.settings import settings
        from dspy.utils.transparency import ACTIVE_CALL_METADATA, ACTIVE_COMPILED_CALL, validate_compiled_call

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
            metadata = ACTIVE_CALL_METADATA.get()
            compiled = resolve_call(
                lm=lm,
                adapter=adapter,
                task_spec=task_spec,
                processed_task_spec=processed_task_spec,
                config=merged_config,
                config_provenance=provenance,
                messages=_history_request_messages_as_openai(request),
                task_spec_mutations=mutations,
                module=metadata.get("module", type(adapter).__name__),
                phase=metadata.get("phase", "adapter"),
                lm_role=metadata.get("lm_role", "default"),
            )
            transparency = settings.get("transparency", "strict")
            validate_compiled_call(compiled, transparency)
            token = ACTIVE_COMPILED_CALL.set(compiled)
            try:
                response = await adapter._call_lm(lm=lm, request=request)
            finally:
                ACTIVE_COMPILED_CALL.reset(token)
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
                config=config,
                task_spec=task_spec,
                demos=demos,
                inputs=inputs,
                error=error,
            )
