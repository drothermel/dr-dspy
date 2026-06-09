from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dspy.history.repl_history import REPLHistory
from dspy.predict.agent_loop import AgentLoopControl, AgentLoopRunner, AgentStepResult
from dspy.predict.predict import Predict
from dspy.predict.rlm import execution as rlm_execution
from dspy.predict.rlm import task_specs as rlm_task_specs
from dspy.predict.rlm import tools as rlm_tools
from dspy.predict.tools import normalize_tools
from dspy.primitives import Module, Prediction
from dspy.runtime.call_options import ModuleCallOptions  # noqa: TC001 — runtime signature typing
from dspy.runtime.run_context import RunContext, resolve_run
from dspy.task_spec import TaskSpec
from dspy.task_spec.framework.rlm import FrameworkRlmSubQueryTaskSpec

if TYPE_CHECKING:
    from dspy.adapters.types.tool import Tool
    from dspy.clients.base_lm import BaseLM
    from dspy.primitives import CodeInterpreter


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
        self._user_tools = normalize_tools(tools)
        rlm_tools.validate_tools(self._user_tools)
        action_sig, extract_sig = rlm_task_specs.build_task_specs(self)
        self.generate_action = Predict(action_sig)
        self.extract = Predict(extract_sig)
        self._sub_query_predict = Predict(FrameworkRlmSubQueryTaskSpec())

    @property
    def tools(self) -> dict[str, Tool]:
        return dict(self._user_tools)

    async def _aforward_impl(
        self,
        *,
        run: RunContext,
        options: ModuleCallOptions | None = None,
        **input_args: Any,
    ) -> Prediction:
        run = resolve_run(run=run, bound_run=self.run)
        rlm_execution.validate_inputs(self, input_args)
        output_field_names = list(self.task_spec.output_fields.keys())
        execution_tools = rlm_execution.prepare_execution_tools(self, run=run)
        variables = rlm_execution.build_variables(self, **input_args)
        with rlm_execution.interpreter_context(self, execution_tools) as repl:
            regular_args = rlm_execution.prepare_serializable_vars(input_args, repl)
            history = REPLHistory(max_output_chars=self.max_output_chars)

            async def step(iteration: int, history: REPLHistory) -> AgentStepResult[REPLHistory]:
                result = await rlm_execution.aexecute_iteration(
                    self,
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
                    return AgentStepResult(
                        history=history,
                        control=AgentLoopControl.RETURN,
                        return_value=result,
                    )
                return AgentStepResult(history=result)

            loop_result = await AgentLoopRunner[REPLHistory]().run(
                max_iters=self.max_iterations,
                initial_history=history,
                step=step,
            )
            if loop_result.return_value is not None:
                return loop_result.return_value
            return await rlm_execution.aextract_fallback(
                self,
                variables,
                loop_result.history,
                output_field_names,
                run=run,
                options=options,
            )
