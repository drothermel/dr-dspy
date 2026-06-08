from dspy.task_spec import FieldSpec, TaskSpec, input_field, output_field


class FrameworkGepaInstructionProposalTaskSpec(TaskSpec):
    name: str = "framework.gepa.instruction_proposal"
    instructions: str = (
        "You are an instruction optimizer. Given the current instruction and dataset examples with feedback, "
        "propose an improved instruction that will help a language model perform the task better."
    )
    inputs: tuple[FieldSpec, ...] = (
        input_field(
            "current_instruction_doc",
            str,
            desc="The current instruction document for the module being optimized.",
        ),
        input_field(
            "dataset_with_feedback",
            str,
            desc="Training examples with inputs, outputs, and feedback for reflection.",
        ),
    )
    outputs: tuple[FieldSpec, ...] = (
        output_field(
            "new_instruction",
            str,
            desc="The improved instruction text to use for the module.",
        ),
    )
