from dspy.task_spec import FieldSpec, TaskSpec, input_field, output_field


class BasicGenerateInstructionTaskSpec(TaskSpec):
    name: str = "framework.copro.basic_generate_instruction"
    instructions: str = "You are an instruction optimizer for large language models. I will give you a ``signature`` of fields (inputs and outputs) in English. Your task is to propose an instruction that will lead a good language model to perform the task well. Don't be afraid to be creative."
    inputs: tuple[FieldSpec, ...] = (
        input_field("basic_instruction", str, desc="The initial instructions before optimization"),
    )
    outputs: tuple[FieldSpec, ...] = (
        output_field("proposed_instruction", str, desc="The improved instructions for the language model"),
        output_field(
            "proposed_prefix_for_output_field",
            str,
            desc="The string at the end of the prompt, which will help the model start solving the task",
        ),
    )


class GenerateInstructionGivenAttemptsTaskSpec(TaskSpec):
    name: str = "framework.copro.generate_instruction_given_attempts"
    instructions: str = "You are an instruction optimizer for large language models. I will give some task instructions I've tried, along with their corresponding validation scores. The instructions are arranged in increasing order based on their scores, where higher scores indicate better quality.\n\nYour task is to propose a new instruction that will lead a good language model to perform the task even better. Don't be afraid to be creative."
    inputs: tuple[FieldSpec, ...] = (
        input_field(
            "attempted_instructions", str, desc="Previously attempted instructions and their validation scores."
        ),
    )
    outputs: tuple[FieldSpec, ...] = (
        output_field("proposed_instruction", str, desc="The improved instructions for the language model"),
        output_field(
            "proposed_prefix_for_output_field",
            str,
            desc="The string at the end of the prompt, which will help the model start solving the task",
        ),
    )
