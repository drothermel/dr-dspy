import logging

from dspy.predict.chain_of_thought import ChainOfThought
from dspy.predict.code_execution import execute_generated_code, parse_generated_code
from dspy.primitives import Module
from dspy.primitives.python_interpreter import PythonInterpreter
from dspy.runtime.call_options import ModuleCallOptions
from dspy.runtime.run_context import RunContext
from dspy.task_spec import FieldSpec, TaskSpec, input_field, make_task_spec, output_field

logger = logging.getLogger(__name__)


class ProgramOfThought(Module):
    def __init__(self, task_spec: TaskSpec, max_iters: int = 3, interpreter: PythonInterpreter | None = None) -> None:
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
        self.interpreter = interpreter or PythonInterpreter()

    def _mode_fields(self, mode: str) -> dict[str, FieldSpec]:
        fields = dict(self.input_fields)
        fields_for_mode = {
            "generate": {
                "generated_code": output_field("generated_code", str, desc="python code that answers the question")
            },
            "regenerate": {
                "previous_code": input_field(
                    "previous_code", str, desc="previously-generated python code that errored"
                ),
                "error": input_field("error", str, desc="error message from previously-generated python code"),
                "generated_code": output_field("generated_code", str, desc="python code that answers the question"),
            },
            "answer": {
                "final_generated_code": input_field(
                    "final_generated_code", str, desc="python code that answers the question"
                ),
                "code_output": input_field("code_output", str, desc="output of previously-generated python code"),
                **self.output_fields,
            },
        }
        fields.update(fields_for_mode[mode])
        return fields

    def _generate_instruction(self, mode):
        mode_fields = self._mode_fields(mode)
        mode_inputs = ", ".join(
            [f"`{field_name}`" for field_name in mode_fields if mode_fields[field_name].role == "input"]
        )
        mode_outputs = ", ".join(
            [f"`{field_name}`" for field_name in mode_fields if mode_fields[field_name].role == "output"]
        )
        final_outputs = ", ".join([f"`{field_name}`" for field_name in self.output_fields])
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
        else:
            instr = [f"Given the final code {mode_inputs}, provide the final {mode_outputs}."]
        return "\n".join(instr)

    async def _aforward_impl(
        self,
        *,
        run: RunContext,
        options: ModuleCallOptions | None = None,
        **inputs,
    ):
        try:
            input_kwargs = {field_name: inputs[field_name] for field_name in self.input_fields if field_name in inputs}
            code_data = await self.code_generate(**input_kwargs, run=run, options=options)
            output = None
            code, error = parse_generated_code(code_data)
            if not error:
                output, error = execute_generated_code(code=code, interpreter=self.interpreter)
            hop = 1
            while error is not None:
                logger.error(f"Error in code execution: {error}")
                if hop == self.max_iters:
                    raise RuntimeError(f"Max hops reached. Failed to run ProgramOfThought: {error}")
                input_kwargs.update({"previous_code": code, "error": error})
                code_data = await self.code_regenerate(**input_kwargs, run=run, options=options)
                code, error = parse_generated_code(code_data)
                if not error:
                    output, error = execute_generated_code(code=code, interpreter=self.interpreter)
                hop += 1
            input_kwargs.update({"final_generated_code": code, "code_output": output})
            return await self.generate_output(**input_kwargs, run=run, options=options)
        finally:
            self.interpreter.shutdown()
