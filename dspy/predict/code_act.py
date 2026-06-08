import inspect
import logging
from collections.abc import Callable

from typing_extensions import override

from dspy.adapters.types.tool import Tool, tool_from_callable
from dspy.predict.chain_of_thought import ChainOfThought
from dspy.predict.predict import Predict
from dspy.predict.program_of_thought import ProgramOfThought
from dspy.predict.react import ReAct
from dspy.primitives.prediction import Prediction
from dspy.primitives.python_interpreter import PythonInterpreter
from dspy.task_spec import TaskSpec, input_field, make_task_spec, output_field

logger = logging.getLogger(__name__)


class CodeAct(ReAct, ProgramOfThought):
    """
    CodeAct is a module that utilizes the Code Interpreter and predefined tools to solve the problem.
    """

    def __init__(
        self,
        task_spec: TaskSpec,
        tools: list[Callable],
        max_iters: int = 5,
        interpreter: PythonInterpreter | None = None,
    ) -> None:
        """
        Initializes the CodeAct class with the specified model, temperature, and max tokens.

        Args:
            task_spec: The task spec of the module.
            tools (list[Callable]): The tool callables to be used. CodeAct only accepts functions and not callable objects.
            max_iters (int): The maximum number of iterations to generate the answer.
            interpreter: PythonInterpreter instance to use. If None, a new one is instantiated.
        Examples:
            ```python
            from dspy.predict import CodeAct
            from dspy.task_spec import TaskSpec, input_field, output_field
            def factorial(n):
                if n == 1:
                    return 1
                return n * factorial(n-1)

            class FactorialTaskSpec(TaskSpec):
                name: str = "Factorial"
                instructions: str = "Compute factorial."
                inputs: tuple = (input_field("n"),)
                outputs: tuple = (output_field("factorial"),)

            act = CodeAct(FactorialTaskSpec(), tools=[factorial])
            act(n=5) # 120
            ```
        """
        if not isinstance(task_spec, TaskSpec):
            raise TypeError(f"CodeAct requires a TaskSpec instance, got {type(task_spec).__name__}.")
        self.task_spec = task_spec
        self.max_iters = max_iters
        self.history = []

        normalized_tools: list[Tool] = [tool_from_callable(t) for t in tools]
        if any(not inspect.isfunction(tool.func) for tool in normalized_tools):
            raise ValueError("CodeAct only accepts functions and not callable objects.")
        tools_by_name: dict[str, Tool] = {}
        for tool in normalized_tools:
            if tool.name is None:
                raise ValueError("Tool name could not be determined.")
            tools_by_name[tool.name] = tool

        instructions = self._build_instructions(task_spec, tools_by_name)

        codeact_task_spec = (
            make_task_spec(
                dict(task_spec.input_fields),
                instructions="\n".join(instructions),
            )
            .append(input_field("trajectory", str))
            .append(
                output_field(
                    "generated_code",
                    str,
                    desc="Python code that when executed, produces output relevant to answering the question",
                ),
            )
            .append(
                output_field(
                    "finished",
                    bool,
                    desc="a boolean flag to determine if the process is done",
                ),
            )
        )

        extract_task_spec = make_task_spec(
            {**task_spec.input_fields, **task_spec.output_fields},
            instructions=task_spec.instructions,
        ).append(input_field("trajectory", str))

        self.tools: dict[str, Tool] = tools_by_name
        self.codeact = Predict(codeact_task_spec)
        self.extractor = ChainOfThought(extract_task_spec)
        # PythonInterpreter may raise if the Deno-backed sandbox is unavailable; construct it here so failures surface during module initialization.
        self.interpreter = interpreter or PythonInterpreter()

    def _build_instructions(self, task_spec, tools):
        instructions = [f"{task_spec.instructions}\n"] if task_spec.instructions else []
        inputs = ", ".join([f"`{k}`" for k in task_spec.input_fields])
        outputs = ", ".join([f"`{k}`" for k in task_spec.output_fields])

        instructions.append(
            f"You are an intelligent agent. For each episode, you will receive the fields {inputs} as input.\n"
            f"Your goal is to generate executable Python code that collects any necessary information for producing {outputs}.\n"
            "For each iteration, you will generate a code snippet that either solves the task or progresses towards the solution.\n"
            "Ensure any output you wish to extract from the code is printed to the console. The code should be enclosed in a fenced code block.\n"
            f"When all information for producing the outputs ({outputs}) are available to be extracted, mark `finished=True` besides the final Python code.\n"
            "You have access to the Python Standard Library and the following functions:"
        )

        for idx, tool in enumerate(tools.values()):
            instructions.append(f"({idx + 1}) {tool}")

        return instructions

    @override
    async def aforward(self, **kwargs):
        for tool in self.tools.values():
            self.interpreter(inspect.getsource(tool.func))

        trajectory = {}
        max_iters = kwargs.pop("max_iters", self.max_iters)
        for idx in range(max_iters):
            code_data = await self.codeact(trajectory=trajectory, **kwargs)
            output = None
            code, error = self._parse_code(code_data)

            if error:
                trajectory[f"observation_{idx}"] = f"Failed to parse the generated code: {error}"
                continue

            trajectory[f"generated_code_{idx}"] = code
            output, error = self._execute_code(code)

            if not error:
                trajectory[f"code_output_{idx}"] = output
            else:
                trajectory[f"observation_{idx}"] = f"Failed to execute the generated code: {error}"

            if code_data.finished:
                break

        extract = await self._call_with_potential_trajectory_truncation(self.extractor, trajectory, **kwargs)
        self.interpreter.shutdown()
        return Prediction(trajectory=trajectory, **extract)
