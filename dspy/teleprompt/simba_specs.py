from dspy.task_spec import FieldSpec, TaskSpec, input_field, output_field


class SimbaOfferFeedbackTaskSpec(TaskSpec):
    name: str = "framework.simba.offer_feedback"
    instructions: str = "You will be given two trajectories of an LLM-driven program's execution. Your goal is to help the program's modules build up experience on how to maximize the reward value assigned to the program's outputs if it were to receive similar inputs in the future.\n\nThe module won't see its own history. It will rely on your advice balancing being concrete and being generalizable.\n\nIn your advice:\n- Avoid boilerplate. Offer advice that would change the module's behavior for the better in the future.\n- Ensure that advice offered to a module M is specific to that M's specific sub-task, not the overall program.\n- Rely on contrasting the behavior of the worse trajectory against the better trajectory in making recommendations.\n- Ensure each unique module name appears exactly once as a key in the advice dictionary."
    inputs: tuple[FieldSpec, ...] = (
        input_field("program_code", str, desc="The code of the program that we are analyzing"),
        input_field("modules_defn", str, desc="The definition of each module in the program, including its I/O"),
        input_field("program_inputs", str, desc="The inputs to the program that we are analyzing"),
        input_field(
            "oracle_metadata", str, desc="Any (hidden) metadata about the training set instance we're analyzing"
        ),
        input_field(
            "worse_program_trajectory", str, desc="The trajectory of the program's execution, showing each module's I/O"
        ),
        input_field("worse_program_outputs", str, desc="The outputs of the program that we are analyzing"),
        input_field("worse_reward_value", float, desc="The reward value assigned to the program's outputs"),
        input_field(
            "worse_reward_info",
            str,
            desc="Additional information that might be helpful to understanding the assigned reward value.",
        ),
        input_field(
            "better_program_trajectory",
            str,
            desc="The trajectory of the program's execution, showing each module's I/O",
        ),
        input_field("better_program_outputs", str, desc="The outputs of the program that we are analyzing"),
        input_field("better_reward_value", float, desc="The reward value assigned to the program's outputs"),
        input_field(
            "better_reward_info",
            str,
            desc="Additional information that might be helpful to understanding the assigned reward value.",
        ),
        input_field(
            "module_names", list[str], desc="The names of the modules in the program, for which we seek advice"
        ),
    )
    outputs: tuple[FieldSpec, ...] = (
        output_field("discussion", str, desc="Discussing blame of where each module went wrong, if it did"),
        output_field(
            "module_advice",
            dict[str, str],
            desc="For each module, describe very concretely: If the module receives ${description of input or patterns therein}, then it should ${description of content, behavior, or strategies to adopt and/or others to avoid}. Basically, your advice be such that if the module has access to your tip, it would be much more likely to act like the successful trajectory rather than the lower-scoring trajectory.",
        ),
    )
