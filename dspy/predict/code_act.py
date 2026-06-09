import logging

from typing_extensions import override

from dspy.adapters.types.tool import Tool
from dspy.core.types.call_options import ModuleCallOptions
from dspy.history import TurnEvent, TurnLog, call_with_turn_log_truncation
from dspy.predict.chain_of_thought import ChainOfThought
from dspy.predict.code_execution import execute_generated_code, parse_generated_code
from dspy.predict.predict import Predict
from dspy.predict.tools import normalize_tools
from dspy.primitives import Module, Prediction
from dspy.primitives.python_interpreter import PythonInterpreter
from dspy.propose.source_format import get_formatted_source
from dspy.runtime.run_context import RunContext
from dspy.task_spec import TaskSpec, input_field, make_task_spec, output_field

logger = logging.getLogger(__name__)


class CodeAct(Module):
    def __init__(
        self, task_spec: TaskSpec, tools: list[Tool], max_iters: int = 5, interpreter: PythonInterpreter | None = None
    ) -> None:
        super().__init__()
        if not isinstance(task_spec, TaskSpec):
            raise TypeError(f"CodeAct requires a TaskSpec instance, got {type(task_spec).__name__}.")
        self.task_spec = task_spec
        self.max_iters = max_iters
        tools_by_name = normalize_tools(tools, require_plain_function=True)
        instructions = self._build_instructions(task_spec, tools_by_name)
        codeact_task_spec = (
            make_task_spec(dict(task_spec.input_fields), instructions="\n".join(instructions))
            .append(input_field("turn_log", TurnLog, desc="Previous code executions and their outputs."))
            .append(
                output_field(
                    "generated_code",
                    str,
                    desc="Python code that when executed, produces output relevant to answering the question",
                )
            )
            .append(output_field("finished", bool, desc="a boolean flag to determine if the process is done"))
        )
        extract_task_spec = make_task_spec(
            {**task_spec.input_fields, **task_spec.output_fields}, instructions=task_spec.instructions
        ).append(input_field("turn_log", TurnLog, desc="Previous code executions and their outputs."))
        self.tools: dict[str, Tool] = tools_by_name
        self.codeact = Predict(codeact_task_spec)
        self.extractor = ChainOfThought(extract_task_spec)
        self.interpreter = interpreter or PythonInterpreter()

    def _build_instructions(self, task_spec, tools):
        instructions = [f"{task_spec.instructions}\n"] if task_spec.instructions else []
        inputs = ", ".join([f"`{k}`" for k in task_spec.input_fields])
        outputs = ", ".join([f"`{k}`" for k in task_spec.output_fields])
        instructions.append(
            f"You are an intelligent agent. For each episode, you will receive the fields {inputs} as input.\nYour goal is to generate executable Python code that collects any necessary information for producing {outputs}.\nFor each iteration, you will generate a code snippet that either solves the task or progresses towards the solution.\nEnsure any output you wish to extract from the code is printed to the console. The code should be enclosed in a fenced code block.\nWhen all information for producing the outputs ({outputs}) are available to be extracted, mark `finished=True` besides the final Python code.\nYou have access to the Python Standard Library and the following functions:"
        )
        for idx, tool in enumerate(tools.values()):
            instructions.append(f"({idx + 1}) {tool}")
        return instructions

    @override
    async def _aforward_impl(
        self,
        *,
        run: RunContext,
        options: ModuleCallOptions | None = None,
        **inputs,
    ):
        try:
            for tool in self.tools.values():
                self.interpreter(get_formatted_source(tool.func))
            turn_log = TurnLog.empty()
            max_iters = inputs.pop("max_iters", self.max_iters)
            for _idx in range(max_iters):
                extracted = await call_with_turn_log_truncation(
                    self.codeact, turn_log=turn_log, run=run, options=options, **inputs
                )
                turn_log = extracted.turn_log
                code_data = extracted.result
                code, error = parse_generated_code(code_data)
                if error:
                    turn_log = turn_log.append_turn(
                        TurnEvent(observation=f"Failed to parse the generated code: {error}")
                    )
                    continue
                output, error = execute_generated_code(code=code, interpreter=self.interpreter)
                event = TurnEvent(generated_code=code)
                if not error:
                    event = event.model_copy(update={"code_output": output})
                else:
                    event = event.model_copy(update={"observation": f"Failed to execute the generated code: {error}"})
                turn_log = turn_log.append_turn(event)
                if code_data.finished:
                    break
            extracted = await call_with_turn_log_truncation(
                self.extractor, turn_log=turn_log, run=run, options=options, **inputs
            )
            return Prediction(turn_log=extracted.turn_log, **dict(extracted.result.items()))
        finally:
            self.interpreter.shutdown()
