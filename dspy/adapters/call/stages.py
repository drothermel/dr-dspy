from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from dspy.core.types import LMRequest, LMResponse  # noqa: TC001
from dspy.core.types.config import LMConfig, coerce_lm_config
from dspy.core.types.openai_compat import request_messages_as_openai
from dspy.runtime.transparency import resolve_call, resolve_call_site, resolve_lm_config, validate_compiled_call
from dspy.task_spec import TaskSpec  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import Mapping

    from dspy.adapters.base.adapter import Adapter
    from dspy.clients.base_lm import BaseLM
    from dspy.runtime.config import CallSite
    from dspy.runtime.run_context import RunContext


class PreparedAdapterCall(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    original_task_spec: TaskSpec
    processed_task_spec: TaskSpec
    request_config: LMConfig
    config_provenance: dict[str, str] = Field(default_factory=dict)
    request: LMRequest
    mutations: list[str] = Field(default_factory=list)


def prepare_adapter_call(
    adapter: Adapter,
    *,
    lm: BaseLM,
    config: LMConfig | Mapping[str, Any] | None,
    task_spec: TaskSpec,
    demos: list[dict[str, Any]],
    inputs: dict[str, Any],
) -> PreparedAdapterCall:
    resolved_config = coerce_lm_config(config)
    original_field_names = set(task_spec.fields.keys())
    processed_task_spec, tools, resolved_config = adapter._call_preprocess(
        lm=lm, config=resolved_config, task_spec=task_spec, inputs=inputs
    )
    mutations = [
        f"removed field {name}" for name in sorted(original_field_names - set(processed_task_spec.fields.keys()))
    ]
    messages = adapter.format(task_spec=processed_task_spec, demos=demos, inputs=inputs)
    request = adapter._render_request(lm=lm, config=resolved_config, tools=tools, messages=messages)
    request_config, provenance = resolve_lm_config(lm, resolved_config)
    return PreparedAdapterCall(
        original_task_spec=task_spec,
        processed_task_spec=processed_task_spec,
        request_config=request_config,
        config_provenance=provenance,
        request=request,
        mutations=mutations,
    )


async def invoke_adapter_lm(
    adapter: Adapter,
    prepared: PreparedAdapterCall,
    *,
    lm: BaseLM,
    run: RunContext,
    call_site: CallSite | None = None,
) -> LMResponse:
    site = resolve_call_site(
        run=run,
        call_site=call_site,
        default_module=type(adapter).__name__,
        default_phase="adapter",
    )
    compiled = resolve_call(
        lm=lm,
        adapter=adapter,
        task_spec=prepared.original_task_spec,
        processed_task_spec=prepared.processed_task_spec,
        config=prepared.request_config,
        config_provenance=prepared.config_provenance,
        messages=request_messages_as_openai(prepared.request),
        task_spec_mutations=prepared.mutations,
        module=site.module,
        phase=site.phase,
        lm_role=site.lm_role,
    )
    validate_compiled_call(compiled, run.telemetry.transparency)
    return await adapter._call_lm(lm=lm, request=prepared.request, run=run, compiled=compiled)
