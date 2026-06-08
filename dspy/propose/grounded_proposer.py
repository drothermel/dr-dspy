import logging
import random

from typing_extensions import override

from dspy.predict.predict import Predict
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.propose.dataset_summary_generator import create_dataset_summary
from dspy.propose.propose_base import Proposer
from dspy.propose.utils import (
    create_example_string,
    create_predictor_level_history_string,
    get_dspy_source_code,
    strip_prefix,
)
from dspy.task_spec import FieldSpec, TaskSpec, input_field, make_task_spec, output_field
from dspy.teleprompt.utils import get_prompt_model, get_task_spec

logger = logging.getLogger(__name__)
MAX_INSTRUCT_IN_HISTORY = 5
TIPS = {
    "none": "",
    "creative": "Don't be afraid to be creative when creating the new instruction!",
    "simple": "Keep the instruction clear and concise.",
    "description": "Make sure your instruction is very informative and descriptive.",
    "high_stakes": "The instruction should include a high stakes scenario in which the LM must solve the task!",
    "persona": 'Include a persona that is relevant to the task in the instruction (ie. "You are a ...")',
}


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
    instructions: str = "Below is some pseudo-code for a pipeline that solves tasks with calls to language models. Please describe the purpose of one of the specified module in this pipeline."
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
        name="GenerateSingleModuleInstruction",
    )


def generate_instruction_class(
    use_dataset_summary=True, program_aware=True, use_task_demos=True, use_instruct_history=True, use_tip=True
):
    return Predict(
        generate_instruction_task_spec(
            use_dataset_summary=use_dataset_summary,
            program_aware=program_aware,
            use_task_demos=use_task_demos,
            use_instruct_history=use_instruct_history,
            use_tip=use_tip,
        )
    )


class GenerateModuleInstruction(Module):
    def __init__(
        self,
        program_code_string=None,
        use_dataset_summary=True,
        program_aware=False,
        use_task_demos=True,
        use_instruct_history=True,
        use_tip=True,
        verbose=False,
    ) -> None:
        super().__init__()
        self.use_dataset_summary = use_dataset_summary
        self.program_aware = program_aware
        self.use_task_demos = use_task_demos
        self.use_instruct_history = use_instruct_history
        self.use_tip = use_tip
        self.verbose = verbose
        self.program_code_string = program_code_string
        self.describe_program = Predict(DescribeProgramTaskSpec())
        self.describe_module = Predict(DescribeModuleTaskSpec())
        self.generate_module_instruction = generate_instruction_class(
            use_dataset_summary=use_dataset_summary,
            program_aware=program_aware,
            use_task_demos=use_task_demos,
            use_instruct_history=use_instruct_history,
            use_tip=use_tip,
        )

    async def aforward(
        self,
        demo_candidates,
        pred_i,
        demo_set_i,
        program,
        previous_instructions,
        data_summary,
        num_demos_in_context=3,
        tip=None,
    ):

        def gather_examples_from_sets(candidate_sets, max_examples):
            count = 0
            for candidate_set in candidate_sets:
                for example in candidate_set:
                    if "augmented" in example:
                        fields_to_use = get_task_spec(program.predictors()[pred_i]).fields
                        yield create_example_string(fields=fields_to_use, example=example)
                        count += 1
                        if count >= max_examples:
                            return

        basic_instruction = get_task_spec(program.predictors()[pred_i]).instructions
        task_demos = ""
        if self.use_task_demos:
            adjacent_sets = (
                [demo_candidates[pred_i][demo_set_i]]
                + demo_candidates[pred_i][demo_set_i + 1 :]
                + demo_candidates[pred_i][:demo_set_i]
            )
            example_strings = gather_examples_from_sets(candidate_sets=adjacent_sets, max_examples=num_demos_in_context)
            task_demos = "\n\n".join(example_strings) + "\n\n"
        if not task_demos.strip() or demo_set_i == 0:
            task_demos = "No task demos provided."
        program_description = "Not available"
        module_code = "Not provided"
        module_description = "Not provided"
        if self.program_aware:
            try:
                program_description = strip_prefix(
                    (
                        await self.describe_program(program_code=self.program_code_string, program_example=task_demos)
                    ).program_description
                )
                if self.verbose:
                    pass
                inputs = []
                outputs = []
                for field_name, field in get_task_spec(program.predictors()[pred_i]).fields.items():
                    if field.role == "input":
                        inputs.append(field_name)
                    else:
                        outputs.append(field_name)
                module_code = (
                    f"{program.predictors()[pred_i].__class__.__name__}({', '.join(inputs)}) -> {', '.join(outputs)}"
                )
                module_description = (
                    await self.describe_module(
                        program_code=self.program_code_string,
                        program_description=program_description,
                        program_example=task_demos,
                        module=module_code,
                        max_depth=10,
                    )
                ).module_description
            except Exception:
                if self.verbose:
                    pass
                self.program_aware = False
        if self.verbose:
            pass
        instruct = await self.generate_module_instruction(
            dataset_description=data_summary,
            program_code=self.program_code_string,
            module=module_code,
            program_description=program_description,
            module_description=module_description,
            task_demos=task_demos,
            tip=tip,
            basic_instruction=basic_instruction,
            previous_instructions=previous_instructions,
        )
        proposed_instruction = strip_prefix(instruct.proposed_instruction)
        return Prediction(proposed_instruction=proposed_instruction)


class GroundedProposer(Proposer):
    def __init__(
        self,
        prompt_model,
        program,
        trainset,
        view_data_batch_size=10,
        use_dataset_summary=True,
        program_aware=True,
        use_task_demos=True,
        num_demos_in_context=3,
        use_instruct_history=True,
        use_tip=True,
        set_tip_randomly=True,
        set_history_randomly=True,
        verbose=False,
        rng=None,
        init_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.program_aware = program_aware
        self.use_dataset_summary = use_dataset_summary
        self.use_task_demos = use_task_demos
        self.num_demos_in_context = num_demos_in_context
        self.use_instruct_history = use_instruct_history
        self.use_tip = use_tip
        self.set_tip_randomly = set_tip_randomly
        self.set_history_randomly = set_history_randomly
        self.verbose = verbose
        self.rng = rng or random
        self.prompt_model = get_prompt_model(prompt_model)
        self.init_temperature = init_temperature
        self.program_code_string = None
        if self.program_aware:
            try:
                self.program_code_string = get_dspy_source_code(program)
            except Exception as exc:
                logger.warning(
                    "Could not extract source code for program-aware instruction proposal; disabling program_aware. Define DSPy programs in .py files. %s",
                    exc,
                )
                self.program_aware = False
        self.data_summary = None
        self._summary_trainset = trainset
        self._view_data_batch_size = view_data_batch_size

    async def _ensure_data_summary(self) -> None:
        if self.data_summary is None and self.use_dataset_summary:
            self.data_summary = await create_dataset_summary(
                trainset=self._summary_trainset,
                view_data_batch_size=self._view_data_batch_size,
                prompt_model=self.prompt_model,
            )

    @override
    async def propose_instructions_for_program(
        self, trainset, program, demo_candidates, trial_logs, N
    ) -> dict[int, list[str]]:
        await self._ensure_data_summary()
        proposed_instructions = {}
        if self.set_history_randomly:
            use_history = self.rng.random() < 0.5
            self.use_instruct_history = use_history
            if self.verbose:
                pass
        if not demo_candidates:
            if self.verbose:
                pass
            self.use_task_demos = False
            num_demos = N
        else:
            num_demos = max(len(demo_candidates[0]), 1)
        for pred_i, predictor in enumerate(program.predictors()):
            for demo_set_i in range(num_demos)[: min(N, num_demos)]:
                if pred_i not in proposed_instructions:
                    proposed_instructions[pred_i] = []
                selected_tip = None
                if self.set_tip_randomly:
                    if self.verbose:
                        pass
                    selected_tip_key = self.rng.choice(list(TIPS.keys()))
                    selected_tip = TIPS[selected_tip_key]
                    self.use_tip = bool(selected_tip)
                    if self.verbose:
                        pass
                proposed_instructions[pred_i].append(
                    await self.propose_instruction_for_predictor(
                        program=program,
                        predictor=predictor,
                        pred_i=pred_i,
                        demo_candidates=demo_candidates,
                        demo_set_i=demo_set_i,
                        trial_logs=trial_logs,
                        tip=selected_tip,
                    )
                )
        return proposed_instructions

    @override
    async def propose_instruction_for_predictor(
        self, program, predictor, pred_i, demo_candidates, demo_set_i, trial_logs, tip=None
    ) -> str:
        instruction_history = create_predictor_level_history_string(
            base_program=program, predictor_i=pred_i, trial_logs=trial_logs, top_n=MAX_INSTRUCT_IN_HISTORY
        )
        instruction_generator = GenerateModuleInstruction(
            program_code_string=self.program_code_string,
            use_dataset_summary=self.use_dataset_summary,
            program_aware=self.program_aware,
            use_task_demos=self.use_task_demos and demo_candidates,
            use_instruct_history=self.use_instruct_history and instruction_history,
            use_tip=self.use_tip,
            verbose=self.verbose,
        )
        rollout_lm = self.prompt_model.copy(temperature=self.init_temperature)
        from dspy.teleprompt.utils import optimizer_lm_context

        with optimizer_lm_context(lm=rollout_lm, phase="propose.grounded", lm_role="prompt_model"):
            proposed_instruction = (
                await instruction_generator(
                    demo_candidates=demo_candidates,
                    pred_i=pred_i,
                    demo_set_i=demo_set_i,
                    program=program,
                    data_summary=self.data_summary,
                    previous_instructions=instruction_history,
                    num_demos_in_context=self.num_demos_in_context,
                    tip=tip,
                )
            ).proposed_instruction
        if self.verbose:
            self.prompt_model.inspect_history(n=1)
        return strip_prefix(proposed_instruction)
