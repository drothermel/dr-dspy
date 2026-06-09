from __future__ import annotations

import base64
import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast

import pydantic

from dspy.adapters.utils import parse_value
from dspy.core.types.call_options import ModuleCallOptions  # noqa: TC001 — runtime signature typing
from dspy.predict.rlm.sync_bridge import _strip_code_fences
from dspy.predict.rlm.tools import make_llm_tools
from dspy.primitives.code_interpreter import SIMPLE_TYPES, CodeInterpreter, CodeInterpreterError, FinalOutput
from dspy.primitives.prediction import Prediction
from dspy.primitives.python_interpreter import PythonInterpreter
from dspy.primitives.repl_types import REPLEntry, REPLHistory, REPLVariable
from dspy.primitives.sandbox_serializable import SandboxSerializable, build_repl_variable
from dspy.task_spec.pydantic_bridge import task_spec_input_field_infos

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from dspy.predict.rlm.module import RLM
    from dspy.runtime.run_context import RunContext
logger = logging.getLogger(__name__)


def get_output_fields_info(rlm: RLM) -> list[dict]:
    fields = []
    for name, field in rlm.task_spec.output_fields.items():
        annotation = field.type_
        field_info = {"name": name}
        if annotation in SIMPLE_TYPES:
            field_info["type"] = annotation.__name__
        fields.append(field_info)
    return fields


def build_variables(rlm: RLM, **input_args: Any) -> list[REPLVariable]:
    variables = []
    input_field_infos = task_spec_input_field_infos(rlm.task_spec)
    for name, value in input_args.items():
        field_info = input_field_infos.get(name)
        if isinstance(value, SandboxSerializable):
            var = build_repl_variable(value, name, field_info=field_info)
        else:
            var = REPLVariable.from_value(name, value, field_info=field_info)
        variables.append(var)
    return variables


def format_output(output: str) -> str:
    if not output:
        return "(no output - did you forget to print?)"
    return output


def validate_inputs(rlm: RLM, input_args: dict[str, Any]) -> None:
    missing = set(rlm.task_spec.input_fields.keys()) - set(input_args.keys())
    if missing:
        raise ValueError(f"Missing required inputs: {sorted(missing)}")


def prepare_serializable_vars(input_args: dict[str, Any], repl: CodeInterpreter) -> dict[str, Any]:
    repl.start()
    regular_args = {}
    for name, value in input_args.items():
        if not isinstance(value, SandboxSerializable):
            regular_args[name] = value
            continue
        payload = value.to_sandbox()
        setup = value.sandbox_setup()
        raw_var_name = f"_raw_{name}"
        assignment = value.sandbox_assignment(name, raw_var_name)
        code_lines = []
        payload_vars: dict[str, str] = {}
        if isinstance(payload, bytes):
            try:
                payload_vars[raw_var_name] = payload.decode("utf-8")
            except UnicodeDecodeError:
                encoded_var_name = f"{raw_var_name}_base64"
                payload_vars[encoded_var_name] = base64.b64encode(payload).decode("ascii")
                code_lines.extend(["import base64", f"{raw_var_name} = base64.b64decode({encoded_var_name})"])
        else:
            payload_vars[raw_var_name] = str(payload)
        if setup:
            code_lines.append(setup)
        code_lines.append(assignment)
        repl.execute("\n".join(code_lines), variables=payload_vars)
    return regular_args


def prepare_execution_tools(rlm: RLM, run: RunContext | None = None) -> dict[str, Callable]:
    execution_tools = make_llm_tools(rlm, run=run)
    execution_tools.update({name: tool.func for name, tool in rlm._user_tools.items()})
    return execution_tools


def inject_execution_context(rlm: RLM, interpreter: CodeInterpreter, execution_tools: dict[str, Callable]) -> None:
    interpreter.tools.update(execution_tools)
    if hasattr(interpreter, "output_fields"):
        cast("Any", interpreter).output_fields = get_output_fields_info(rlm)
    if hasattr(interpreter, "_tools_registered"):
        cast("Any", interpreter)._tools_registered = False


@contextmanager
def interpreter_context(rlm: RLM, execution_tools: dict[str, Callable]) -> Iterator[CodeInterpreter]:
    if rlm._interpreter is not None:
        inject_execution_context(rlm, rlm._interpreter, execution_tools)
        yield rlm._interpreter
    else:
        repl = PythonInterpreter(tools=execution_tools, output_fields=get_output_fields_info(rlm))
        try:
            yield repl
        finally:
            repl.shutdown()


def extract_fallback(
    rlm: RLM, variables: list[REPLVariable], history: REPLHistory, output_field_names: list[str]
) -> Prediction:
    logger.warning("RLM reached max iterations, using extract to get final output")
    variables_info = [variable.format() for variable in variables]
    extract_pred = rlm.extract(variables_info=variables_info, turn_log=history)
    return Prediction(
        turn_log=history,
        final_reasoning="Extract forced final output",
        **{name: getattr(extract_pred, name) for name in output_field_names},
    )


async def aextract_fallback(
    rlm: RLM,
    variables: list[REPLVariable],
    history: REPLHistory,
    output_field_names: list[str],
    *,
    run: RunContext,
    options: ModuleCallOptions | None = None,
) -> Prediction:
    logger.warning("RLM reached max iterations, using extract to get final output")
    variables_info = [variable.format() for variable in variables]
    extract_pred = await rlm.extract(
        variables_info=variables_info,
        turn_log=history,
        run=run,
        options=options,
    )
    return Prediction(
        turn_log=history,
        final_reasoning="Extract forced final output",
        **{name: getattr(extract_pred, name) for name in output_field_names},
    )


def process_final_output(
    rlm: RLM, result: FinalOutput, output_field_names: list[str]
) -> tuple[dict[str, Any] | None, str | None]:
    raw_output = result.output
    if not isinstance(raw_output, dict):
        return (
            None,
            f"[Error] FINAL returned {type(raw_output).__name__}, expected dict with fields: {output_field_names}",
        )
    missing = set(output_field_names) - set(raw_output.keys())
    if missing:
        return (None, f"[Error] Missing output fields: {sorted(missing)}. Use SUBMIT({', '.join(output_field_names)})")
    parsed_outputs = {}
    type_errors = []
    for name in output_field_names:
        field = rlm.task_spec.output_fields[name]
        annotation = field.type_
        try:
            parsed_outputs[name] = parse_value(raw_output[name], annotation)
        except (ValueError, pydantic.ValidationError) as e:
            type_errors.append(
                f"{name}: expected {(annotation.__name__ if hasattr(annotation, '__name__') else annotation)}, got {type(raw_output[name]).__name__}: {e}"
            )
    if type_errors:
        return (None, "[Type Error] " + "; ".join(type_errors))
    return (parsed_outputs, None)


def process_execution_result(
    rlm: RLM, pred: Prediction, code: str, result: Any, history: REPLHistory, output_field_names: list[str]
) -> Prediction | REPLHistory:
    if isinstance(result, str) and result.startswith("[Error]"):
        output = format_output(result)
        return history.append(reasoning=pred.reasoning, code=code, output=output)
    if isinstance(result, FinalOutput):
        parsed_outputs, error = process_final_output(rlm, result, output_field_names)
        if error:
            return history.append(reasoning=pred.reasoning, code=code, output=error)
        final_history = history.append(reasoning=pred.reasoning, code=code, output=f"FINAL: {parsed_outputs}")
        return Prediction(**parsed_outputs or {}, turn_log=final_history, final_reasoning=pred.reasoning)
    output = "\n".join(map(str, result)) if isinstance(result, list) else str(result) if result else ""
    output = format_output(output)
    if rlm.verbose:
        logger.info(REPLEntry.format_output(output, rlm.max_output_chars))
    return history.append(reasoning=pred.reasoning, code=code, output=output)


def execute_code(repl: CodeInterpreter, code: str, input_args: dict[str, Any]) -> Any:
    try:
        return repl.execute(code, variables=dict(input_args))
    except (CodeInterpreterError, SyntaxError) as e:
        return f"[Error] {e}"


def execute_iteration(
    rlm: RLM,
    repl: CodeInterpreter,
    variables: list[REPLVariable],
    history: REPLHistory,
    iteration: int,
    input_args: dict[str, Any],
    output_field_names: list[str],
) -> Prediction | REPLHistory:
    variables_info = [variable.format() for variable in variables]
    action = rlm.generate_action(
        variables_info=variables_info, turn_log=history, iteration=f"{iteration + 1}/{rlm.max_iterations}"
    )
    if rlm.verbose:
        logger.info(
            f"RLM iteration {iteration + 1}/{rlm.max_iterations}\nReasoning: {action.reasoning}\nCode:\n{action.code}"
        )
    try:
        code = _strip_code_fences(action.code)
    except SyntaxError as e:
        code = action.code
        result = f"[Error] {e}"
        return process_execution_result(rlm, action, code, result, history, output_field_names)
    result = execute_code(repl, code, input_args)
    return process_execution_result(rlm, action, code, result, history, output_field_names)


async def aexecute_iteration(
    rlm: RLM,
    repl: CodeInterpreter,
    variables: list[REPLVariable],
    history: REPLHistory,
    iteration: int,
    input_args: dict[str, Any],
    output_field_names: list[str],
    *,
    run: RunContext,
    options: ModuleCallOptions | None = None,
) -> Prediction | REPLHistory:
    variables_info = [variable.format() for variable in variables]
    pred = await rlm.generate_action(
        variables_info=variables_info,
        turn_log=history,
        iteration=f"{iteration + 1}/{rlm.max_iterations}",
        run=run,
        options=options,
    )
    if rlm.verbose:
        logger.info(
            f"RLM iteration {iteration + 1}/{rlm.max_iterations}\nReasoning: {pred.reasoning}\nCode:\n{pred.code}"
        )
    try:
        code = _strip_code_fences(pred.code)
    except SyntaxError as e:
        code = pred.code
        result = f"[Error] {e}"
        return process_execution_result(rlm, pred, code, result, history, output_field_names)
    result = execute_code(repl, code, input_args)
    return process_execution_result(rlm, pred, code, result, history, output_field_names)
