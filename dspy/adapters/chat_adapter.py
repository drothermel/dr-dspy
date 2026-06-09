from __future__ import annotations

from typing import TYPE_CHECKING, Any

from typing_extensions import override

from dspy.adapters.base import Adapter
from dspy.adapters.call.capabilities import AdapterCapabilities
from dspy.adapters.call.policies.json_parse_fallback import JSONParseFallbackPolicy
from dspy.adapters.format_shared import FIELD_HEADER_PATTERN, ChatFormatMixin
from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.utils import parse_output_field, validate_parsed_fields

if TYPE_CHECKING:
    from dspy.adapters.call.policies.parse_fallback import NoOpParseFallbackPolicy
    from dspy.adapters.types.base_type import Type
    from dspy.runtime.callback import Callback
    from dspy.task_spec import TaskSpec

__all__ = ["ChatAdapter"]

_DEFAULT_PARSE_FALLBACK = object()


class ChatAdapter(ChatFormatMixin, Adapter):
    capabilities = AdapterCapabilities(
        supports_finetune=True,
        field_value_role="none",
        default_native_fc=False,
        supports_structured_output=False,
    )

    def __init__(
        self,
        callbacks: list[Callback] | None = None,
        use_native_function_calling: bool = False,
        native_response_types: list[type[Type]] | None = None,
        parallel_tool_calls: bool | None = None,
        json_fallback: JSONAdapter | None = None,
        parse_fallback_policy: JSONParseFallbackPolicy | NoOpParseFallbackPolicy | None = None,
    ) -> None:
        super().__init__(
            callbacks=callbacks,
            use_native_function_calling=use_native_function_calling,
            parallel_tool_calls=parallel_tool_calls,
            native_response_types=native_response_types,
        )
        self._json_fallback = json_fallback
        if parse_fallback_policy is None:
            self.parse_fallback_policy = JSONParseFallbackPolicy(fallback_factory=self._json_adapter_fallback)
        else:
            self.parse_fallback_policy = parse_fallback_policy

    def _json_adapter_fallback(self) -> JSONAdapter:
        if self._json_fallback is not None:
            return self._json_fallback
        return JSONAdapter(
            callbacks=self.callbacks,
            use_native_function_calling=self.use_native_function_calling,
            parallel_tool_calls=self.parallel_tool_calls,
            native_response_types=self.native_response_types,
        )

    @override
    def parse(self, task_spec: TaskSpec, completion: str) -> dict[str, Any]:
        sections = [(None, [])]
        for line in completion.splitlines():
            match = FIELD_HEADER_PATTERN.match(line.strip())
            if match:
                header = match.group(1)
                remaining_content = line[match.end() :].strip()
                sections.append((header, [remaining_content] if remaining_content else []))
            else:
                sections[-1][1].append(line)
        sections = [(k, "\n".join(v).strip()) for k, v in sections]
        fields = {}
        for k, v in sections:
            if k not in fields and k in task_spec.output_fields:
                fields[k] = parse_output_field(
                    adapter_name="ChatAdapter",
                    task_spec=task_spec,
                    field_name=k,
                    raw_value=v,
                    lm_response=completion,
                    field=task_spec.output_fields[k],
                )
        validate_parsed_fields(adapter_name="ChatAdapter", task_spec=task_spec, lm_response=completion, fields=fields)
        return fields
