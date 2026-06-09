from dspy.task_spec import FieldSpec, TaskSpec, input_field, make_task_spec, output_field


class DescribeProgramTaskSpec(TaskSpec):
    name: str = "framework.propose.describe_program"
    instructions: str = "Below is some pseudo-code for a pipeline that solves tasks with calls to language models. Please describe what type of task this program appears to be designed to solve, and how it appears to work."
    inputs: tuple[FieldSpec, ...] = (
        input_field(
            "program_code", str, desc="Pseudocode for a language model program designed to solve a particular task."
        ),
        input_field("program_example", str, desc="An example of the program in use."),
    )
    outputs: tuple[FieldSpec, ...] = (
        output_field(
            "program_description",
            str,
            desc="Describe what task the program is designed to solve, and how it goes about solving this task.",
        ),
    )


class DescribeModuleTaskSpec(TaskSpec):
    name: str = "framework.propose.describe_module"
    instructions: str = "Below is some pseudo-code for a pipeline that solves tasks with calls to language models. Please describe the purpose of the specified module in this pipeline."
    inputs: tuple[FieldSpec, ...] = (
        input_field(
            "program_code", str, desc="Pseudocode for a language model program designed to solve a particular task."
        ),
        input_field("program_example", str, desc="An example of the program in use."),
        input_field(
            "program_description",
            str,
            desc="Summary of the task the program is designed to solve, and how it goes about solving it.",
        ),
        input_field("module", str, desc="The module in the program that we want to describe."),
    )
    outputs: tuple[FieldSpec, ...] = (
        output_field("module_description", str, desc="Description of the module's role in the broader program."),
    )


class ObservationSummarizerTaskSpec(TaskSpec):
    name: str = "framework.propose.observation_summarizer"
    instructions: str = "Given a series of observations I have made about my dataset, please summarize them into a brief 2-3 sentence summary which highlights only the most important details."
    inputs: tuple[FieldSpec, ...] = (
        input_field("observations", str, desc="Observations I have made about my dataset"),
    )
    outputs: tuple[FieldSpec, ...] = (
        output_field(
            "summary",
            str,
            desc="Two-to-three sentence summary of only the most significant highlights of my observations",
        ),
    )


class DatasetDescriptorTaskSpec(TaskSpec):
    name: str = "framework.propose.dataset_descriptor"
    instructions: str = "Given several examples from a dataset please write observations about trends that hold for most or all of the samples. Some areas you may consider in your observations: topics, content, syntax, conciseness, etc. It will be useful to make an educated guess as to the nature of the task this dataset will enable. Don't be afraid to be creative"
    inputs: tuple[FieldSpec, ...] = (input_field("examples", str, desc="Sample data points from the dataset"),)
    outputs: tuple[FieldSpec, ...] = (
        output_field(
            "observations",
            str,
            desc="Something that holds true for most or all of the data you observed",
        ),
    )


class DatasetDescriptorWithPriorObservationsTaskSpec(TaskSpec):
    name: str = "framework.propose.dataset_descriptor_with_prior"
    instructions: str = "Given several examples from a dataset please write observations about trends that hold for most or all of the samples. I will also provide you with a few observations I have already made. Please add your own observations or if you feel the observations are comprehensive say 'COMPLETE'. Some areas you may consider in your observations: topics, content, syntax, conciseness, etc. It will be useful to make an educated guess as to the nature of the task this dataset will enable. Don't be afraid to be creative"
    inputs: tuple[FieldSpec, ...] = (
        input_field("examples", str, desc="Sample data points from the dataset"),
        input_field("prior_observations", str, desc="Some prior observations I made about the data"),
    )
    outputs: tuple[FieldSpec, ...] = (
        output_field(
            "observations",
            str,
            desc="Something that holds true for most or all of the data you observed or COMPLETE if you have nothing to add",
        ),
    )


def generate_instruction_task_spec(
    use_dataset_summary=True, program_aware=True, use_task_demos=True, use_instruct_history=True, use_tip=True
):
    fields: dict[str, FieldSpec] = {}
    if use_dataset_summary:
        fields["dataset_description"] = input_field(
            "dataset_description", str, desc="A description of the dataset that we are using."
        )
    if program_aware:
        fields["program_code"] = input_field(
            "program_code", str, desc="Language model program designed to solve a particular task."
        )
        fields["program_description"] = input_field(
            "program_description",
            str,
            desc="Summary of the task the program is designed to solve, and how it goes about solving it.",
        )
        fields["module"] = input_field("module", str, desc="The module to create an instruction for.")
        fields["module_description"] = input_field(
            "module_description", str, desc="Description of the module to create an instruction for."
        )
    if use_task_demos:
        fields["task_demos"] = input_field("task_demos", str, desc="Example inputs/outputs of our module.")
    if use_instruct_history:
        fields["previous_instructions"] = input_field(
            "previous_instructions",
            str,
            desc="Previous instructions we've attempted, along with their associated scores.",
        )
    fields["basic_instruction"] = input_field("basic_instruction", str, desc="Basic instruction.")
    if use_tip:
        fields["tip"] = input_field("tip", str, desc="A suggestion for how to go about generating the new instruction.")
    fields["proposed_instruction"] = output_field(
        "proposed_instruction",
        str,
        desc="Propose an instruction that will be used to prompt a Language Model to perform this task.",
    )
    return make_task_spec(
        fields,
        instructions="Use the information below to learn about a task that we are trying to solve using calls to an LM, then generate a new instruction that will be used to prompt a Language Model to better solve the task.",
        name="framework.propose.generate_single_module_instruction",
    )
