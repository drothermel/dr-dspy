from __future__ import annotations

import logging
from typing import Any, get_args

import pydantic

from dspy.adapters.types.reasoning import Reasoning
from dspy.adapters.types.tool import Tool, ToolCallResults, ToolCalls
from dspy.core.types.call_options import ModuleCallOptions, PredictOptions
from dspy.core.types.config import LMConfig, LMToolChoice
from dspy.errors import AdapterParseError
from dspy.history import TruncationExhaustedError, TurnEvent, TurnLog, call_with_turn_log_truncation
from dspy.predict.agent_helpers import format_tool_exception
from dspy.predict.predict import Predict
from dspy.predict.tools import normalize_tools
from dspy.primitives import Module, Prediction
from dspy.runtime.run_context import RunContext, resolve_run
from dspy.task_spec import FieldSpec, TaskSpec, input_field, make_task_spec, output_field

logger = logging.getLogger(__name__)


class ReActV2(Module):
    def __init__(self, task_spec: TaskSpec, tools: list[Tool], max_iters: int = 20) -> None:
        super().__init__()
        if not isinstance(task_spec, TaskSpec):
            raise TypeError(f"ReActV2 requires a TaskSpec instance, got {type(task_spec).__name__}.")
        self.task_spec = task_spec
        self.max_iters = max_iters
        self.tools = normalize_tools(tools)
        if "submit" in self.tools:
            raise ValueError("`submit` is reserved by ReActV2 as the final-output tool.")
        self.tools["submit"] = self._make_submit_tool()
        self.react = Predict(self._make_react_task_spec())

    def _make_submit_tool(self) -> Tool:
        output_fields = self.task_spec.output_fields
        output_names = list(output_fields)

        def submit(**kwargs):
            missing = [name for name in output_names if name not in kwargs]
            if missing:
                raise ValueError(f"Missing required final output field(s): {', '.join(missing)}")
            return {name: kwargs[name] for name in output_names}

        args = {name: _json_schema_for_annotation(field.type_) for name, field in output_fields.items()}
        arg_types = {name: field.type_ for name, field in output_fields.items()}
        return Tool(
            submit, description="Submit the final outputs for the task.", name="submit", args=args, arg_types=arg_types
        )

    def _make_react_task_spec(self) -> TaskSpec:
        fields: dict[str, FieldSpec] = {}
        for name, field in self.task_spec.input_fields.items():
            fields[name] = input_field(name, _optional_annotation(field.type_), desc=field.desc)
        fields["turn_log"] = input_field("turn_log", TurnLog, desc="Previous thoughts, tool calls, and tool results.")
        fields["tools"] = input_field("tools", list[Tool], desc="Tools available for this step.")
        fields["next_thought"] = output_field("next_thought", Reasoning, desc="Your next reasoning step.")
        fields["tool_calls"] = output_field("tool_calls", ToolCalls, desc="Tool calls to execute next.")
        inputs = ", ".join(f"`{name}`" for name in self.task_spec.input_fields)
        outputs = ", ".join(f"`{name}`" for name in self.task_spec.output_fields)
        tool_names = ", ".join(f"`{name}`" for name in self.tools)
        instructions = "\n".join(
            [
                self.task_spec.instructions,
                f"You are an Agent. Use the supplied tools to produce {outputs} from {inputs}.",
                "Call tools when more information is needed.",
                f"When the final answer is ready, call `submit` with {outputs}.",
                f"The available tools are: {tool_names}.",
            ]
        ).strip()
        return make_task_spec(fields, instructions=instructions)

    async def _aforward_impl(
        self,
        *,
        run: RunContext,
        options: ModuleCallOptions | None = None,
        **input_args,
    ):
        run = resolve_run(run=run, bound_run=self.run)
        max_iters = input_args.pop("max_iters", self.max_iters)
        if "history" in input_args:
            raise ValueError("ReActV2 accepts `turn_log=` only; the `history=` keyword was removed.")
        turn_log_raw = input_args.pop("turn_log", None)
        turn_log = TurnLog.empty() if turn_log_raw is None else TurnLog.model_validate(turn_log_raw)
        pending_inputs = {name: input_args[name] for name in self.task_spec.input_fields if name in input_args}
        break_reason = "max_iters"
        for turn_index in range(max_iters):
            try:
                extracted = await call_with_turn_log_truncation(
                    self.react,
                    turn_log=turn_log,
                    tools=list(self.tools.values()),
                    **pending_inputs,
                    run=run,
                    options=options,
                )
                pred = extracted.result
                turn_log = extracted.turn_log
                tool_calls = _coerce_tool_calls(getattr(pred, "tool_calls", None))
            except TruncationExhaustedError as err:
                logger.warning("Ending ReActV2 loop after context window exceeded: %s", err)
                break_reason = "context_window_exceeded"
                break
            except ValueError as err:
                logger.warning("Ending ReActV2 loop after parse failure: %s", format_tool_exception(err))
                break_reason = "parse_error"
                break
            except AdapterParseError as err:
                logger.warning("Ending ReActV2 loop after parse failure: %s", format_tool_exception(err))
                break_reason = "parse_error"
                break
            if not tool_calls.tool_calls:
                break_reason = "empty_tool_calls"
                break
            tool_calls = _ensure_tool_call_ids(tool_calls, turn_index)
            tool_call_results, final_outputs = await self._execute_tool_calls(tool_calls)
            event = self._history_event(pending_inputs, pred, tool_calls, tool_call_results)
            if final_outputs is not None:
                event.update(final_outputs)
            turn_log = turn_log.append_turn(TurnEvent.model_validate(event))
            pending_inputs = {}
            if final_outputs is not None:
                return Prediction(**final_outputs, turn_log=turn_log, termination_reason="submit")
        return await self._forced_submit(turn_log, pending_inputs, break_reason, max_iters, run=run)

    async def _execute_tool_calls(self, tool_calls: ToolCalls) -> tuple[ToolCallResults, dict[str, Any] | None]:
        values = []
        is_errors = []
        final_outputs = None
        for tool_call in tool_calls.tool_calls:
            if tool_call.name not in self.tools:
                values.append(f"Unknown tool: {tool_call.name}")
                is_errors.append(True)
                continue
            try:
                value = await self.tools[tool_call.name].acall(**(tool_call.args or {}))
                values.append(value)
                is_errors.append(False)
                if tool_call.name == "submit" and isinstance(value, dict):
                    final_outputs = dict(value)
            except Exception as err:
                values.append(f"Execution error in {tool_call.name}: {format_tool_exception(err)}")
                is_errors.append(True)
        return (ToolCallResults.from_tool_calls_and_values(tool_calls, values, is_errors), final_outputs)

    def _history_event(
        self,
        pending_inputs: dict[str, Any],
        pred: Prediction,
        tool_calls: ToolCalls,
        tool_call_results: ToolCallResults,
    ) -> dict[str, Any]:
        event = dict(pending_inputs)
        if hasattr(pred, "next_thought") and pred.next_thought is not None:
            event["next_thought"] = pred.next_thought
        if tool_calls.tool_calls:
            if tool_call_results.tool_call_results:
                tool_calls = tool_calls.model_copy(update={"tool_call_results": tool_call_results})
            event["tool_calls"] = tool_calls
        return event

    async def _forced_submit(
        self,
        turn_log: TurnLog,
        pending_inputs: dict[str, Any],
        break_reason: str,
        turn_index: int,
        *,
        run: RunContext,
    ) -> Prediction:
        try:
            extracted = await call_with_turn_log_truncation(
                self.react,
                turn_log=turn_log,
                tools=list(self.tools.values()),
                options=PredictOptions(
                    config=LMConfig(
                        tool_choice=LMToolChoice(mode="required", allowed=["submit"]),
                        reasoning=None,
                    )
                ),
                **pending_inputs,
                run=run,
            )
            pred = extracted.result
            turn_log = extracted.turn_log
            tool_calls = _ensure_tool_call_ids(_coerce_tool_calls(getattr(pred, "tool_calls", None)), turn_index)
        except TruncationExhaustedError as err:
            logger.warning("Forced submit failed after context window exceeded: %s", err)
            return Prediction(turn_log=turn_log, termination_reason=break_reason or "failed")
        except ValueError as err:
            logger.warning("Forced submit failed: %s", format_tool_exception(err))
            return Prediction(turn_log=turn_log, termination_reason=break_reason or "failed")
        except AdapterParseError as err:
            logger.warning("Forced submit failed: %s", format_tool_exception(err))
            return Prediction(turn_log=turn_log, termination_reason=break_reason or "failed")
        submit_calls = ToolCalls(tool_calls=[call for call in tool_calls.tool_calls if call.name == "submit"])
        if not submit_calls.tool_calls:
            return Prediction(turn_log=turn_log, termination_reason=break_reason or "failed")
        tool_call_results, final_outputs = await self._execute_tool_calls(submit_calls)
        event = self._history_event(pending_inputs, pred, submit_calls, tool_call_results)
        if final_outputs is not None:
            event.update(final_outputs)
        turn_log = turn_log.append_turn(TurnEvent.model_validate(event))
        if final_outputs is not None:
            return Prediction(**final_outputs, turn_log=turn_log, termination_reason="forced_submit")
        return Prediction(turn_log=turn_log, termination_reason=break_reason or "failed")


def _json_schema_for_annotation(annotation: Any) -> dict[str, Any]:
    try:
        return pydantic.TypeAdapter(annotation).json_schema()
    except Exception:
        return {"type": "string"}


def _optional_annotation(annotation: Any) -> Any:
    if type(None) in get_args(annotation):
        return annotation
    try:
        return annotation | None
    except TypeError:
        return annotation


def _coerce_tool_calls(tool_calls: Any) -> ToolCalls:
    if tool_calls is None:
        return ToolCalls(tool_calls=[])
    return ToolCalls.model_validate(tool_calls)


def _ensure_tool_call_ids(tool_calls: ToolCalls, turn_index: int) -> ToolCalls:
    ensured = []
    for call_index, tool_call in enumerate(tool_calls.tool_calls):
        if tool_call.id is None:
            tool_call = tool_call.model_copy(update={"id": f"call_{turn_index}_{call_index}"})
        ensured.append(tool_call)
    return ToolCalls(tool_calls=ensured)
