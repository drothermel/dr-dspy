"""Adapter wrappers that delegate surface behavior to an inner adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dspy.adapters.base import Adapter
from dspy.adapters.call.pipeline import AdapterCallPipeline
from dspy.task_spec import TaskSpec, input_field

if TYPE_CHECKING:
    from dspy.core.types import LMMessage, UserMessageContent
    from dspy.task_spec import FieldBinding

if TYPE_CHECKING:
    from collections.abc import Mapping

    from dspy.clients.base_lm import BaseLM
    from dspy.core.types import LMConfig
    from dspy.runtime.config import CallSite
    from dspy.runtime.run_context import RunContext


class AdapterWrapper(Adapter):
    def __init__(self, inner: Adapter) -> None:
        super().__init__(
            callbacks=inner.callbacks,
            use_native_function_calling=inner.use_native_function_calling,
            native_response_types=inner.native_response_types,
            parallel_tool_calls=inner.parallel_tool_calls,
            allow_json_repair=inner.allow_json_repair,
        )
        self._inner = inner
        self._sync_from_inner()

    def _sync_from_inner(self) -> None:
        self.response_format_policy = self._inner.response_format_policy
        self.parse_fallback_policy = self._inner.parse_fallback_policy
        self.capabilities = self._inner.capabilities
        self.field_formatter = self._inner.field_formatter
        self.preprocessor_chain = self._inner.preprocessor_chain
        self.message_assembler = self._inner.message_assembler

    def format(self, task_spec: TaskSpec, demos: list[dict[str, Any]], inputs: dict[str, Any]) -> list[LMMessage]:
        return self._inner.format(task_spec=task_spec, demos=demos, inputs=inputs)

    def parse(self, task_spec: TaskSpec, completion: str) -> dict[str, Any]:
        return self._inner.parse(task_spec=task_spec, completion=completion)

    def format_system_message(self, task_spec: TaskSpec) -> str:
        return self._inner.format_system_message(task_spec)

    def format_field_description(self, task_spec: TaskSpec) -> str:
        return self._inner.format_field_description(task_spec)

    def format_field_structure(self, task_spec: TaskSpec) -> str:
        return self._inner.format_field_structure(task_spec)

    def format_task_description(self, task_spec: TaskSpec) -> str:
        return self._inner.format_task_description(task_spec)

    def format_user_message_content(
        self,
        task_spec: TaskSpec,
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> UserMessageContent:
        return self._inner.format_user_message_content(
            task_spec=task_spec,
            inputs=inputs,
            prefix=prefix,
            suffix=suffix,
            main_request=main_request,
        )

    def format_assistant_message_content(
        self, task_spec: TaskSpec, outputs: dict[str, Any], missing_field_message: str | None = None
    ) -> str:
        return self._inner.format_assistant_message_content(
            task_spec=task_spec,
            outputs=outputs,
            missing_field_message=missing_field_message,
        )

    def format_demos(self, task_spec: TaskSpec, demos: list[dict[str, Any]]) -> list[LMMessage]:
        return self._inner.format_demos(task_spec=task_spec, demos=demos)

    def format_finetune_data(
        self,
        task_spec: TaskSpec,
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
        outputs: dict[str, Any],
    ) -> dict[str, list[Any]]:
        return self._inner.format_finetune_data(task_spec=task_spec, demos=demos, inputs=inputs, outputs=outputs)

    def format_field_with_value(
        self,
        fields_with_values: dict[FieldBinding, Any],
        *,
        role_label: str | None = None,
        **kwargs: Any,
    ) -> str:
        return self._inner.format_field_with_value(fields_with_values, role_label=role_label, **kwargs)


class HintInjectingAdapter(AdapterWrapper):
    """Augments task inputs with ``hint_`` and delegates execution to ``_inner``."""

    def __init__(self, inner: Adapter, hint_map: dict[str, str], task_spec_to_name: dict[TaskSpec, str]) -> None:
        super().__init__(inner)
        self._hint_map = hint_map
        self._task_spec_to_name = task_spec_to_name

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
        self._sync_from_inner()
        hint_name = self._task_spec_to_name.get(task_spec, "N/A")
        inputs = dict(inputs)
        inputs["hint_"] = self._hint_map.get(hint_name, "N/A")
        hinted_task_spec = task_spec.append(
            input_field("hint_", str, desc="A hint to the module from an earlier run"),
        )
        return await AdapterCallPipeline.execute(
            self._inner,
            lm=lm,
            config=config,
            task_spec=hinted_task_spec,
            demos=demos,
            inputs=inputs,
            run=run,
            call_site=call_site,
        )
