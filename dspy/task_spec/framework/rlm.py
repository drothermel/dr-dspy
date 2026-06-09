from dspy.task_spec import TaskSpec, input_field, output_field


class FrameworkRlmSubQueryTaskSpec(TaskSpec):
    name: str = "framework.rlm.sub_query"
    instructions: str = "Answer the prompt concisely and directly."
    inputs: tuple = (input_field("prompt", str, desc="The sub-LLM query prompt to answer."),)
    outputs: tuple = (output_field("response", str, desc="The sub-LLM response text."),)
