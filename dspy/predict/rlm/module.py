"""
Recursive Language Model (RLM) module for DSPy.

RLMs are an inference strategy where LLMs treat long contexts as part of an external
environment rather than feeding them directly to the model. The LLM writes Python code
to programmatically examine, decompose, and recursively call sub-LLMs over snippets.

Reference: "Recursive Language Models" (Zhang, Kraska, Khattab, 2025)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dspy.predict.predict import Predict
from dspy.predict.rlm import execution as rlm_execution
from dspy.predict.rlm import task_specs as rlm_task_specs
from dspy.predict.rlm import tools as rlm_tools
from dspy.predict.rlm.task_specs import FrameworkRlmSubQueryTaskSpec
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.primitives.repl_types import REPLHistory, REPLVariable
from dspy.task_spec import TaskSpec
from dspy.utils.annotation import experimental

if TYPE_CHECKING:
    from collections.abc import Callable

    from dspy.adapters.types.tool import Tool
    from dspy.clients.base_lm import BaseLM
    from dspy.primitives.code_interpreter import CodeInterpreter


@experimental
class RLM(Module):
    """Recursive Language Model module.

    Uses a sandboxed REPL to let the LLM programmatically explore large contexts
    through code execution. The LLM writes Python code to examine data, call
    sub-LLMs for semantic analysis, and build up answers iteratively.

    The default interpreter is PythonInterpreter (Deno/Pyodide/WASM), but you
    can provide any CodeInterpreter implementation (e.g., MockInterpreter, or write a custom one using E2B or Modal).

    Note: RLM instances are not thread-safe when using a custom interpreter.
    Create separate RLM instances for concurrent use, or use the default
    PythonInterpreter which creates a fresh instance per forward() call.

    Examples:
        ```python
        # Basic usage
        rlm = dspy.RLM("context, query -> output", max_iterations=10)
        result = rlm(context="...very long text...", query="What is the magic number?")
        print(result.output)
        ```
    """

    def __init__(
        self,
        task_spec: TaskSpec,
        max_iterations: int = 20,
        max_llm_calls: int = 50,
        max_output_chars: int = 10_000,
        verbose: bool = False,
        tools: list[Callable] | None = None,
        sub_lm: BaseLM | None = None,
        interpreter: CodeInterpreter | None = None,
    ) -> None:
        """
        Args:
            task_spec: Defines inputs and outputs as a TaskSpec instance.
            max_iterations: Maximum REPL interaction iterations.
            max_llm_calls: Maximum sub-LLM calls (llm_query/llm_query_batched) per execution.
            max_output_chars: Maximum characters to include from REPL output.
            verbose: Whether to log detailed execution info.
            tools: List of tool functions or dspy.adapters.types.tool.Tool objects callable from interpreter code.
                  Built-in tools: llm_query(prompt), llm_query_batched(prompts).
            sub_lm: LM for llm_query/llm_query_batched. Defaults to dspy.settings.lm.
                   Allows using a different (e.g., cheaper) model for sub-queries.
            interpreter: CodeInterpreter implementation to use. Defaults to PythonInterpreter.
        """
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

        # Build the action and extract signatures
        action_sig, extract_sig = rlm_task_specs.build_task_specs(self)
        self.generate_action = Predict(action_sig)
        self.extract = Predict(extract_sig)
        self._sub_query_predict = Predict(FrameworkRlmSubQueryTaskSpec())

    def _normalize_tools(self, tools_list: list[Callable] | None) -> dict[str, Tool]:
        return rlm_tools.normalize_tools(tools_list)

    def _validate_tools(self, user_tools: dict[str, Tool]) -> None:
        rlm_tools.validate_tools(user_tools)

    def _format_tool_docs(self, user_tools: dict[str, Tool]) -> str:
        return rlm_tools.format_tool_docs(user_tools)

    def _make_llm_tools(self, max_workers: int = 8) -> dict[str, Callable]:
        return rlm_tools.make_llm_tools(self, max_workers=max_workers)

    @property
    def tools(self) -> dict[str, Tool]:
        """User-provided tools (excludes internal llm_query/llm_query_batched)."""
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

    def _prepare_serializable_vars(
        self,
        input_args: dict[str, Any],
        repl: CodeInterpreter,
    ) -> dict[str, Any]:
        return rlm_execution.prepare_serializable_vars(input_args, repl)

    def _prepare_execution_tools(self) -> dict[str, Callable]:
        return rlm_execution.prepare_execution_tools(self)

    def _inject_execution_context(self, interpreter: CodeInterpreter, execution_tools: dict[str, Callable]) -> None:
        rlm_execution.inject_execution_context(self, interpreter, execution_tools)

    def _interpreter_context(self, execution_tools: dict[str, Callable]):
        return rlm_execution.interpreter_context(self, execution_tools)

    def _extract_fallback(
        self,
        variables: list[REPLVariable],
        history: REPLHistory,
        output_field_names: list[str],
    ) -> Prediction:
        return rlm_execution.extract_fallback(self, variables, history, output_field_names)

    def _process_final_output(
        self,
        result: Any,
        output_field_names: list[str],
    ) -> tuple[dict[str, Any] | None, str | None]:
        return rlm_execution.process_final_output(self, result, output_field_names)

    def _process_execution_result(
        self,
        pred: Prediction,
        code: str,
        result: Any,
        history: REPLHistory,
        output_field_names: list[str],
    ) -> Prediction | REPLHistory:
        return rlm_execution.process_execution_result(self, pred, code, result, history, output_field_names)

    def _execute_code(
        self,
        repl: CodeInterpreter,
        code: str,
        input_args: dict[str, Any],
    ) -> Any:
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
    ) -> Prediction:
        return await rlm_execution.aextract_fallback(self, variables, history, output_field_names)

    async def _aexecute_iteration(
        self,
        repl: CodeInterpreter,
        variables: list[REPLVariable],
        history: REPLHistory,
        iteration: int,
        input_args: dict[str, Any],
        output_field_names: list[str],
    ) -> Prediction | REPLHistory:
        return await rlm_execution.aexecute_iteration(
            self, repl, variables, history, iteration, input_args, output_field_names
        )

    async def aforward(self, **input_args) -> Prediction:
        """Execute RLM to produce outputs from the given inputs.

        Args:
            **input_args: Input values matching the signature's input fields

        Returns:
            Prediction with output field(s) from the signature and 'trajectory' for debugging

        Raises:
            ValueError: If required input fields are missing
        """
        self._validate_inputs(input_args)

        output_field_names = list(self.task_spec.output_fields.keys())
        execution_tools = self._prepare_execution_tools()
        variables = self._build_variables(**input_args)

        with self._interpreter_context(execution_tools) as repl:
            regular_args = self._prepare_serializable_vars(input_args, repl)
            history = REPLHistory(max_output_chars=self.max_output_chars)

            for iteration in range(self.max_iterations):
                result = await self._aexecute_iteration(
                    repl, variables, history, iteration, regular_args, output_field_names
                )
                if isinstance(result, Prediction):
                    return result
                history = result

            # Max iterations reached - use extract fallback
            return await self._aextract_fallback(variables, history, output_field_names)
