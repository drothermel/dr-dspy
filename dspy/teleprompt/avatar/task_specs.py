from pydantic import BaseModel

from dspy.predict.avatar.models import ActionOutput
from dspy.task_spec import FieldSpec, TaskSpec, input_field, output_field


class EvalResult(BaseModel):
    example: dict
    score: float
    actions: list[ActionOutput] | None = None


class ComparatorTaskSpec(TaskSpec):
    name: str = "framework.avatar.comparator"
    instructions: str = "After executing the given actions on user inputs using the given instruction, some inputs have yielded good, results, while others have not. I'll provide you the inputs along with their, corresponding evaluation metrics:\n\nTask:\n(1) Firstly, identify and contrast the patterns of inputs that have achieved good results with those that have not.\n(2) Then, review the computational logic for any inconsistencies in the previous actions.\n(3) Lastly, specify the modification in tools used that can lead to improved performance on the negative inputs."
    inputs: tuple[FieldSpec, ...] = (
        input_field("instruction", str, desc="Instruction for the actor to execute the task"),
        input_field("actions", list[str], desc="Actions actor can take to complete the task"),
        input_field(
            "pos_input_with_metrics",
            list[EvalResult],
            desc="Positive inputs along with their score on a evaluation metric and actions taken",
        ),
        input_field(
            "neg_input_with_metrics",
            list[EvalResult],
            desc="Negative inputs along with their score on a evaluation metric and actions taken",
        ),
    )
    outputs: tuple[FieldSpec, ...] = (
        output_field("feedback", str, desc="Feedback for the actor to improve the performance of negative inputs"),
    )


class FeedbackBasedInstructionTaskSpec(TaskSpec):
    name: str = "framework.avatar.feedback_instruction"
    instructions: str = "There is a task that needs to be completed for which one can use multiple tools to achieve the desired outcome. A group's performance was evaluated on a dataset of inputs, the inputs that did well are positive inputs, and the inputs that did not do well are negative inputs.\n\nYou received feedback on how they can better use the tools to improve your performance on the negative inputs. You have been provided with the previous instruction, that they followed to use tools to complete the task, and the feedback on your performance.\n\nYour task is to incorporate the feedback and generate a detailed instruction for the group to follow to improve their performance on the task.\n\nMake sure that the new instruction talks about how to use the tools effectively and should be no more than 3 paragraphs long. The previous instruction contains general guidelines that you must retain in the new instruction."
    inputs: tuple[FieldSpec, ...] = (
        input_field("previous_instruction", str, desc="Previous instruction for the actor to execute the task"),
        input_field("feedback", str, desc="Feedback for the actor to improve the performance of negative inputs"),
    )
    outputs: tuple[FieldSpec, ...] = (
        output_field("new_instruction", str, desc="New instruction for the actor to execute the task"),
    )
