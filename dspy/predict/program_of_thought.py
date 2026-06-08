import json
import logging
import re
from typing import Any, cast

from dspy.predict.chain_of_thought import ChainOfThought
from dspy.primitives.code_interpreter import FinalOutput
from dspy.primitives.module import Module
from dspy.primitives.python_interpreter import PythonInterpreter
from dspy.signatures.field import InputField, OutputField
from dspy.signatures.signature import Signature, _field_infos_to_signature_fields, ensure_signature, make_signature

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

    lm = LM('openai/gpt-4o-mini')
    settings.configure(lm=lm)
    pot = ProgramOfThought("question -> answer")
    pot(question="what is 1+1?")
    ```
    """

    def __init__(
        self, signature: str | type[Signature], max_iters: int = 3, interpreter: PythonInterpreter | None = None
    ) -> None:
        """
        Args:
            signature: The signature of the module.
            max_iters: The maximum number of iterations to retry code generation and execution.
            interpreter: PythonInterpreter instance to use. If None, a new one is instantiated.
        """
        super().__init__()
        resolved_signature = ensure_signature(signature)
        if resolved_signature is None:
            raise ValueError(f"Invalid signature: {signature!r}")
        self.signature: type[Signature] = resolved_signature
        self.max_iters = max_iters

        self.input_fields = resolved_signature.input_fields
        self.output_fields = resolved_signature.output_fields

        self.code_generate = ChainOfThought(
            make_signature(
                signature=cast("Any", _field_infos_to_signature_fields(self._generate_signature("generate").fields)),
                instructions=self._generate_instruction("generate"),
            ),
        )
        self.code_regenerate = ChainOfThought(
            make_signature(
                signature=cast("Any", _field_infos_to_signature_fields(self._generate_signature("regenerate").fields)),
                instructions=self._generate_instruction("regenerate"),
            ),
        )
        self.generate_output = ChainOfThought(
            make_signature(
                signature=cast("Any", _field_infos_to_signature_fields(self._generate_signature("answer").fields)),
                instructions=self._generate_instruction("answer"),
            ),
        )
        # PythonInterpreter may raise if the Deno-backed sandbox is unavailable; construct it here so failures surface during module initialization.
        self.interpreter = interpreter or PythonInterpreter()

    def _generate_signature(self, mode):
        signature_dict = dict(self.input_fields)
        fields_for_mode = {
            "generate": {
                "generated_code": OutputField(
                    desc="python code that answers the question",
                ),
            },
            "regenerate": {
                "previous_code": InputField(
                    desc="previously-generated python code that errored",
                ),
                "error": InputField(
                    desc="error message from previously-generated python code",
                ),
                "generated_code": OutputField(
                    desc="python code that answers the question",
                ),
            },
            "answer": {
                "final_generated_code": InputField(
                    desc="python code that answers the question",
                ),
                "code_output": InputField(
                    desc="output of previously-generated python code",
                ),
            }
            | self.signature.output_fields,
        }
        signature_dict.update(fields_for_mode[mode])
        return make_signature(cast("Any", _field_infos_to_signature_fields(signature_dict)))

    def _generate_instruction(self, mode):
        mode_inputs = ", ".join(
            [f"`{field_name}`" for field_name in self._generate_signature(mode).input_fields],
        )
        mode_outputs = ", ".join(
            [f"`{field_name}`" for field_name in self._generate_signature(mode).output_fields],
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
            # Serialize interpreter results before passing them back through the answer signature.
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
