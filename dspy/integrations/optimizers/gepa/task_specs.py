from dspy.task_spec import FieldSpec, TaskSpec, input_field, output_field


class GenerateEnhancedMultimodalInstructionTaskSpec(TaskSpec):
    name: str = "framework.gepa.multimodal_instruction"
    instructions: str = "I provided an assistant with instructions to perform a task involving visual content, but the assistant's performance needs improvement based on the examples and feedback below.\n\nYour task is to write a better instruction for the assistant that addresses the specific issues identified in the feedback, with particular attention to how visual and textual information should be analyzed and integrated.\n\n## Analysis Steps:\n1. **Read the inputs carefully** and identify both the visual and textual input formats, understanding how they work together\n2. **Read all the assistant responses and corresponding feedback** to understand what went wrong with visual analysis, text processing, or their integration\n3. **Identify visual analysis patterns** - what visual features, relationships, or details are important for this task\n4. **Identify domain-specific knowledge** about both visual and textual aspects, as this information may not be available to the assistant in the future\n5. **Look for successful visual-textual integration strategies** and include these patterns in the instruction\n6. **Address specific visual analysis issues** mentioned in the feedback\n\n## Instruction Requirements:\n- **Clear task definition** explaining how to process both visual and textual inputs\n- **Visual analysis guidance** specific to this task (what to look for, how to describe, what features matter)\n- **Integration strategies** for combining visual observations with textual information\n- **Domain-specific knowledge** about visual concepts, terminology, or relationships\n- **Error prevention guidance** for common visual analysis mistakes shown in the feedback\n- **Precise, actionable language** for both visual and textual processing\n\nFocus on creating an instruction that helps the assistant properly analyze visual content, integrate it with textual information, and avoid the specific visual analysis mistakes shown in the examples."
    inputs: tuple[FieldSpec, ...] = (
        input_field(
            "current_instruction",
            str,
            desc="The current instruction that was provided to the assistant to perform the multimodal task",
        ),
        input_field(
            "examples_with_feedback",
            str,
            desc="Task examples with visual content showing inputs, assistant outputs, and feedback. Pay special attention to feedback about visual analysis accuracy, visual-textual integration, and any domain-specific visual knowledge that the assistant missed.",
        ),
    )
    outputs: tuple[FieldSpec, ...] = (
        output_field(
            "improved_instruction",
            str,
            desc="A better instruction for the assistant that addresses visual analysis issues, provides clear guidance on how to process and integrate visual and textual information, includes necessary visual domain knowledge, and prevents the visual analysis mistakes shown in the examples.",
        ),
    )


class FrameworkGepaInstructionProposalTaskSpec(TaskSpec):
    name: str = "framework.gepa.instruction_proposal"
    instructions: str = "You are an instruction optimizer. Given the current instruction and dataset examples with feedback, propose an improved instruction that will help a language model perform the task better."
    inputs: tuple[FieldSpec, ...] = (
        input_field(
            "current_instruction_doc", str, desc="The current instruction document for the module being optimized."
        ),
        input_field(
            "dataset_with_feedback", str, desc="Training examples with inputs, outputs, and feedback for reflection."
        ),
    )
    outputs: tuple[FieldSpec, ...] = (
        output_field("new_instruction", str, desc="The improved instruction text to use for the module."),
    )
