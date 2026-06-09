from dspy.task_spec import FieldSpec, TaskSpec, input_field, output_field


class OfferFeedbackTaskSpec(TaskSpec):
    name: str = "framework.refine.offer_feedback"
    instructions: str = "In the discussion, assign blame to each module that contributed to the final reward being below the threshold, if any. Then, prescribe concrete advice of how the module should act on its future input when we retry the process, if it were to receive the same or similar inputs. If a module is not to blame, the advice should be N/A. The module will not see its own history, so it needs to rely on entirely concrete and actionable advice from you to avoid the same mistake on the same or similar inputs."
    inputs: tuple[FieldSpec, ...] = (
        input_field("program_code", str, desc="The code of the program that we are analyzing"),
        input_field("modules_defn", str, desc="The definition of each module in the program, including its I/O"),
        input_field("program_inputs", str, desc="The inputs to the program that we are analyzing"),
        input_field(
            "program_trajectory", str, desc="The trajectory of the program's execution, showing each module's I/O"
        ),
        input_field("program_outputs", str, desc="The outputs of the program that we are analyzing"),
        input_field("reward_code", str, desc="The code of the reward function that we are analyzing"),
        input_field("target_threshold", float, desc="The target threshold for the reward function"),
        input_field("reward_value", float, desc="The reward value assigned to the program's outputs"),
        input_field(
            "module_names", list[str], desc="The names of the modules in the program, for which we seek advice"
        ),
    )
    outputs: tuple[FieldSpec, ...] = (
        output_field("discussion", str, desc="Discussing blame of where each module went wrong, if it did"),
        output_field(
            "advice",
            dict[str, str],
            desc="For each module, describe very concretely, in this order: the specific scenarios in which it has made mistakes in the past and what each mistake was, followed by what it should do differently in that kind of scenario in the future. If the module is not to blame, write N/A.",
        ),
    )
