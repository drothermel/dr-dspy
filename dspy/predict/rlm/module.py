from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dspy.core.types.call_options import ModuleCallOptions  # noqa: TC001 — runtime signature typing
from dspy.history.repl_history import REPLHistory, REPLVariable
from dspy.predict.predict import Predict
from dspy.predict.rlm import execution as rlm_execution
from dspy.predict.rlm import task_specs as rlm_task_specs
from dspy.predict.rlm import tools as rlm_tools
from dspy.predict.rlm.task_specs import FrameworkRlmSubQueryTaskSpec
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.runtime.run_context import RunContext, resolve_run
from dspy.task_spec import TaskSpec

if TYPE_CHECKING:
    from collections.abc import Callable

    from dspy.adapters.types.tool import Tool
    from dspy.clients.base_lm import BaseLM
    from dspy.primitives.code_interpreter import CodeInterpreter


class RLM(Module):
    def __init__(
        self,
        task_spec: TaskSpec,
        max_iterations: int = 20,
        max_llm_calls: int = 50,
        max_output_chars: int = 10000,
        verbose: bool = False,
        tools: list[Tool] | None = None,
        sub_lm: BaseLM | None = None,
        interpreter: CodeInterpreter | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(task_spec, TaskSpec):
            raise TypeError(f"RLM requires a TaskSpec instance, got {type(task_spec).__name__}.")
        self.task_spec = task_spec
        self.max_iterations = max_iterations
        self.max_llm_calls = max_llm_calls
        self.max_output_chars = max_output_chars
        self.verbose = verbose
        self.sub_lm = sub_lm
        self._interpreter = interpreter
        self._user_tools = rlm_tools.normalize_tools(tools)
        rlm_tools.validate_tools(self._user_tools)
        action_sig, extract_sig = rlm_task_specs.build_task_specs(self)
        self.generate_action = Predict(action_sig)
        self.extract = Predict(extract_sig)
        self._sub_query_predict = Predict(FrameworkRlmSubQueryTaskSpec())

    def _normalize_tools(self, tools_list: list[Tool] | None) -> dict[str, Tool]:
        return rlm_tools.normalize_tools(tools_list)

    def _validate_tools(self, user_tools: dict[str, Tool]) -> None:
        rlm_tools.validate_tools(user_tools)

    def _format_tool_docs(self, user_tools: dict[str, Tool]) -> str:
        return rlm_tools.format_tool_docs(user_tools)

    def _make_llm_tools(self, run: RunContext | None = None, max_workers: int = 8) -> dict[str, Callable]:
        return rlm_tools.make_llm_tools(self, run=run, max_workers=max_workers)

    @property
    def tools(self) -> dict[str, Tool]:
        return dict(self._user_tools)

    def _build_task_specs(self) -> tuple[TaskSpec, TaskSpec]:
        return rlm_task_specs.build_task_specs(self)

    def _get_output_fields_info(self) -> list[dict]:
        return rlm_execution.get_output_fields_info(self)

    def _build_variables(self, **input_args: Any) -> list[REPLVariable]:
        return rlm_execution.build_variables(self, **input_args)

    def _format_output(self, output: str) -> str:
        return rlm_execution.format_output(output)

    def _validate_inputs(self, input_args: dict[str, Any]) -> None:
        rlm_execution.validate_inputs(self, input_args)

    def _prepare_serializable_vars(self, input_args: dict[str, Any], repl: CodeInterpreter) -> dict[str, Any]:
        return rlm_execution.prepare_serializable_vars(input_args, repl)

    def _prepare_execution_tools(self, run=None) -> dict[str, Callable]:
        return rlm_execution.prepare_execution_tools(self, run=run)

    def _inject_execution_context(self, interpreter: CodeInterpreter, execution_tools: dict[str, Callable]) -> None:
        rlm_execution.inject_execution_context(self, interpreter, execution_tools)

    def _interpreter_context(self, execution_tools: dict[str, Callable]):
        return rlm_execution.interpreter_context(self, execution_tools)

    def _extract_fallback(
        self, variables: list[REPLVariable], history: REPLHistory, output_field_names: list[str]
    ) -> Prediction:
        return rlm_execution.extract_fallback(self, variables, history, output_field_names)

    def _process_final_output(
        self, result: Any, output_field_names: list[str]
    ) -> tuple[dict[str, Any] | None, str | None]:
        return rlm_execution.process_final_output(self, result, output_field_names)

    def _process_execution_result(
        self, pred: Prediction, code: str, result: Any, history: REPLHistory, output_field_names: list[str]
    ) -> Prediction | REPLHistory:
        return rlm_execution.process_execution_result(self, pred, code, result, history, output_field_names)

    def _execute_code(self, repl: CodeInterpreter, code: str, input_args: dict[str, Any]) -> Any:
        return rlm_execution.execute_code(repl, code, input_args)

    def _execute_iteration(
        self,
        repl: CodeInterpreter,
        variables: list[REPLVariable],
        history: REPLHistory,
        iteration: int,
        input_args: dict[str, Any],
        output_field_names: list[str],
    ) -> Prediction | REPLHistory:
        return rlm_execution.execute_iteration(
            self, repl, variables, history, iteration, input_args, output_field_names
        )

    async def _aextract_fallback(
        self,
        variables: list[REPLVariable],
        history: REPLHistory,
        output_field_names: list[str],
        *,
        run: RunContext,
        options: ModuleCallOptions | None = None,
    ) -> Prediction:
        return await rlm_execution.aextract_fallback(
            self, variables, history, output_field_names, run=run, options=options
        )

    async def _aexecute_iteration(
        self,
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
        return await rlm_execution.aexecute_iteration(
            self,
            repl,
            variables,
            history,
            iteration,
            input_args,
            output_field_names,
            run=run,
            options=options,
        )

    async def aforward(
        self,
        *,
        run: RunContext,
        options: ModuleCallOptions | None = None,
        **input_args,
    ) -> Prediction:
        run = resolve_run(run=run, bound_run=self.run)
        self._validate_inputs(input_args)
        output_field_names = list(self.task_spec.output_fields.keys())
        execution_tools = self._prepare_execution_tools(run=run)
        variables = self._build_variables(**input_args)
        with self._interpreter_context(execution_tools) as repl:
            regular_args = self._prepare_serializable_vars(input_args, repl)
            history = REPLHistory(max_output_chars=self.max_output_chars)
            for iteration in range(self.max_iterations):
                result = await self._aexecute_iteration(
                    repl,
                    variables,
                    history,
                    iteration,
                    regular_args,
                    output_field_names,
                    run=run,
                    options=options,
                )
                if isinstance(result, Prediction):
                    return result
                history = result
            return await self._aextract_fallback(variables, history, output_field_names, run=run, options=options)
