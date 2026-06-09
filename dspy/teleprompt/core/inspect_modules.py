import textwrap

from dspy.adapters.prompt_format import get_field_spec_description_string
from dspy.primitives import Module
from dspy.task_spec.predictor_context import get_task_spec


def inspect_modules(program: Module) -> str:
    separator = "-" * 80
    output = [separator]
    for name, predictor in program.named_predictors():
        task_spec = get_task_spec(predictor)
        instructions = textwrap.dedent(task_spec.instructions)
        instructions = ("\n" + "\t" * 2).join([""] + instructions.splitlines())
        output.append(f"Module {name}")
        output.append("\n\tInput Fields:")
        output.append(
            ("\n" + "\t" * 2).join([""] + get_field_spec_description_string(task_spec.input_fields).splitlines())
        )
        output.append("\tOutput Fields:")
        output.append(
            ("\n" + "\t" * 2).join([""] + get_field_spec_description_string(task_spec.output_fields).splitlines())
        )
        output.append(f"\tOriginal Instructions: {instructions}")
        output.append(separator)
    return "\n".join([o.strip("\n") for o in output])
