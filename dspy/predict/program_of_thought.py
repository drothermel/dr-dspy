import json
import logging
import re

from dspy.predict.chain_of_thought import ChainOfThought
from dspy.primitives.code_interpreter import FinalOutput
from dspy.primitives.module import Module
from dspy.primitives.python_interpreter import PythonInterpreter
from dspy.task_spec import FieldSpec, TaskSpec, make_task_spec

logger = logging.getLogger(__name__)


class ProgramOfThought(Module):
    """
    A DSPy module that runs Python programs to solve a problem.
    This module requires deno to be installed. Please install deno following https://docs.deno.com/runtime/getting_started/installation/

    Examples:
    ```
    from dspy.clients.lm import LM
    from dspy.dsp.utils.settings import settings
    from dspy.predict.program_of_thought import ProgramOfThought
    from dspy.task_spec import make_task_spec

    lm = LM('openai/gpt-4o-mini')
    settings.configure(lm=lm)
    pot = ProgramOfThought(make_task_spec("question -> answer", instructions="Answer the question."))
    pot(question="what is 1+1?")
    ```
    """

    def __init__(self, task_spec: TaskSpec, max_iters: int = 3, interpreter: PythonInterpreter | None = None) -> None:
        """
        Args:
            task_spec: The task spec of the module.
            max_iters: The maximum number of iterations to retry code generation and execution.
            interpreter: PythonInterpreter instance to use. If None, a new one is instantiated.
        """
        super().__init__()
        if not isinstance(task_spec, TaskSpec):
            raise TypeError(f"ProgramOfThought requires a TaskSpec instance, got {type(task_spec).__name__}.")
        self.task_spec = task_spec
        self.max_iters = max_iters

        self.input_fields = task_spec.input_fields
        self.output_fields = task_spec.output_fields

        self.code_generate = ChainOfThought(
            make_task_spec(self._mode_fields("generate"), instructions=self._generate_instruction("generate"))
        )
        self.code_regenerate = ChainOfThought(
            make_task_spec(self._mode_fields("regenerate"), instructions=self._generate_instruction("regenerate"))
        )
        self.generate_output = ChainOfThought(
            make_task_spec(self._mode_fields("answer"), instructions=self._generate_instruction("answer"))
        )
        # PythonInterpreter may raise if the Deno-backed sandbox is unavailable; construct it here so failures surface during module initialization.
        self.interpreter = interpreter or PythonInterpreter()

    def _mode_fields(self, mode: str) -> dict[str, FieldSpec]:
        fields = dict(self.input_fields)
        fields_for_mode = {
            "generate": {
                "generated_code": FieldSpec.output(
                    "generated_code",
                    str,
                    desc="python code that answers the question",
                ),
            },
            "regenerate": {
                "previous_code": FieldSpec.input(
                    "previous_code",
                    str,
                    desc="previously-generated python code that errored",
                ),
                "error": FieldSpec.input(
                    "error",
                    str,
                    desc="error message from previously-generated python code",
                ),
                "generated_code": FieldSpec.output(
                    "generated_code",
                    str,
                    desc="python code that answers the question",
                ),
            },
            "answer": {
                "final_generated_code": FieldSpec.input(
                    "final_generated_code",
                    str,
                    desc="python code that answers the question",
                ),
                "code_output": FieldSpec.input(
                    "code_output",
                    str,
                    desc="output of previously-generated python code",
                ),
                **self.output_fields,
            },
        }
        fields.update(fields_for_mode[mode])
        return fields

    def _generate_instruction(self, mode):
        mode_fields = self._mode_fields(mode)
        mode_inputs = ", ".join(
            [f"`{field_name}`" for field_name in mode_fields if mode_fields[field_name].role == "input"],
        )
        mode_outputs = ", ".join(
            [f"`{field_name}`" for field_name in mode_fields if mode_fields[field_name].role == "output"],
        )
        final_outputs = ", ".join(
            [f"`{field_name}`" for field_name in self.output_fields],
        )
        if mode == "generate":
            instr = [
                f"You will be given {mode_inputs} and you will respond with {mode_outputs}.",
                f"Generating executable Python code that programmatically computes the correct {mode_outputs}.",
                "After you're done with the computation and think you have the final output, make sure to submit your output by calling the preloaded function `SUBMIT()`.",
                f'You must structure your output in a dict, like {{"field_a": value_a, ...}}, with the correct value mapping for the field(s): {final_outputs}.',
            ]
        elif mode == "regenerate":
            instr = [
                f"You are given {mode_inputs} due to an error in previous code.",
                "Your task is to correct the error and provide the new `generated_code`.",
            ]
        else:  # mode == 'answer'
            instr = [
                f"Given the final code {mode_inputs}, provide the final {mode_outputs}.",
            ]

        return "\n".join(instr)

    def _parse_code(self, code_data):
        code = code_data.get("generated_code", "").split("---", 1)[0].split("\n\n\n", 1)[0]
        code_match = re.search(r"```python[ \n](.*?)[ \n]```?", code, re.DOTALL)
        code_block = code_match.group(1) if code_match else code
        if not code_block:
            return code, "Error: Empty code after parsing."
        if "\n" not in code_block and code_block.count("=") > 1:
            return code, "Error: Code format is not correct."
        lines = code_block.split("\n")
        last_line_match = re.match(r"^(\w+)\s*=", lines[-1].strip())
        if last_line_match and len(lines) > 1:
            code_block += "\n" + last_line_match.group(1)
        return code_block, None

    def _execute_code(self, code):
        """
        Execute the code using PythonInterpreter and return the output or error.
        """
        if not code:
            return None, "Error: Empty code before execution."

        try:
            result = self.interpreter.execute(code)
            if isinstance(result, FinalOutput):
                result = result.output
            # Serialize interpreter results before passing them back through the answer task spec.
            output = json.dumps(result)
            return output, None
        except Exception as e:
            return None, str(e)

    async def aforward(self, **kwargs):
        input_kwargs = {field_name: kwargs[field_name] for field_name in self.input_fields}
        code_data = await self.code_generate(**input_kwargs)
        output = None
        code, error = self._parse_code(code_data)
        if not error:
            output, error = self._execute_code(code)
        hop = 1
        while error is not None:
            logger.error(f"Error in code execution: {error}")
            if hop == self.max_iters:
                self.interpreter.shutdown()
                raise RuntimeError(f"Max hops reached. Failed to run ProgramOfThought: {error}")
            input_kwargs.update({"previous_code": code, "error": error})
            code_data = await self.code_regenerate(**input_kwargs)
            code, error = self._parse_code(code_data)
            if not error:
                output, error = self._execute_code(code)
            hop += 1
        input_kwargs.update({"final_generated_code": code, "code_output": output})
        output_gen_result = await self.generate_output(**input_kwargs)
        self.interpreter.shutdown()
        return output_gen_result
