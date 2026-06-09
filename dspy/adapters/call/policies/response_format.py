from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

from dspy.core.types.config import LMConfig, coerce_lm_config
from dspy.utils.exceptions import LMError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from dspy.adapters.base.adapter import Adapter
    from dspy.clients.base_lm import BaseLM
    from dspy.task_spec import TaskSpec

logger = logging.getLogger(__name__)


class ResponseFormatPolicy(Protocol):
    async def execute(
        self,
        *,
        adapter: Adapter,
        lm: BaseLM,
        config: LMConfig | None,
        task_spec: TaskSpec,
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
        run_once: Callable[
            [LMConfig | None],
            Awaitable[list[dict[str, Any]]],
        ],
    ) -> list[dict[str, Any]]: ...


class NoOpResponseFormatPolicy:
    async def execute(
        self,
        *,
        adapter: Adapter,
        lm: BaseLM,
        config: LMConfig | None,
        task_spec: TaskSpec,
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
        run_once: Callable[[LMConfig | None], Awaitable[list[dict[str, Any]]]],
    ) -> list[dict[str, Any]]:
        _ = (adapter, lm, task_spec, demos, inputs)
        return await run_once(config)


class StructuredOutputPolicy:
    async def execute(
        self,
        *,
        adapter: Adapter,
        lm: BaseLM,
        config: LMConfig | None,
        task_spec: TaskSpec,
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
        run_once: Callable[[LMConfig | None], Awaitable[list[dict[str, Any]]]],
    ) -> list[dict[str, Any]]:
        from dspy.adapters.json_adapter import _get_structured_outputs_response_format, _has_open_ended_mapping
        from dspy.adapters.types.tool import ToolCalls

        resolved_config = coerce_lm_config(config)
        if "response_format" not in lm.supported_params:
            return await run_once(resolved_config)

        has_tool_calls = any(field.type_ == ToolCalls for field in task_spec.output_fields.values())
        if (
            _has_open_ended_mapping(task_spec)
            or (not adapter.use_native_function_calling and has_tool_calls)
            or (not lm.supports_response_schema)
        ):
            json_config = resolved_config.model_copy(update={"response_format": {"type": "json_object"}})
            return await run_once(json_config)

        try:
            structured_output_model = _get_structured_outputs_response_format(
                task_spec=task_spec, use_native_function_calling=adapter.use_native_function_calling
            )
            structured_config = resolved_config.model_copy(update={"response_format": structured_output_model})
            return await run_once(structured_config)
        except LMError:
            raise
        except Exception:
            logger.warning("Failed to use structured output format, falling back to JSON mode.")
            json_config = resolved_config.model_copy(update={"response_format": {"type": "json_object"}})
            return await run_once(json_config)
