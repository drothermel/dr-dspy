"""Grounded instruction proposer for teleprompt optimizers.

Import ``GroundedProposer`` from ``dspy.propose.grounded_proposer``.
"""

import logging
import random

from dspy.predict.predict import Predict
from dspy.primitives import Module, Prediction
from dspy.propose.dataset_summary_generator import create_dataset_summary
from dspy.propose.protocol import TrialLogs
from dspy.propose.task_specs import (
    DescribeModuleTaskSpec,
    DescribeProgramTaskSpec,
    generate_instruction_task_spec,
)
from dspy.propose.utils import (
    create_example_string,
    create_predictor_level_history_string,
    get_dspy_source_code,
    strip_prefix,
)
from dspy.teleprompt.task_spec_context import get_task_spec, resolve_optimizer_lm
from dspy.teleprompt.utils import optimizer_lm_context

logger = logging.getLogger(__name__)

__all__ = ["GroundedProposer"]

MAX_INSTRUCT_IN_HISTORY = 5
PROGRAM_AWARE_INPUT_KEYS = frozenset({"program_code", "program_description", "module", "module_description"})
TIPS = {
    "none": "",
    "creative": "Don't be afraid to be creative when creating the new instruction!",
    "simple": "Keep the instruction clear and concise.",
    "description": "Make sure your instruction is very informative and descriptive.",
    "high_stakes": "The instruction should include a high stakes scenario in which the LM must solve the task!",
    "persona": 'Include a persona that is relevant to the task in the instruction (ie. "You are a ...")',
}


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

    async def _aforward_impl(
        self,
        demo_candidates,
        pred_i,
        demo_set_i,
        program,
        previous_instructions,
        data_summary,
        num_demos_in_context=3,
        tip=None,
        *,
        run,
        options=None,
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
        if not task_demos.strip():
            task_demos = "No task demos provided."
        program_description = "Not available"
        module_code = "Not provided"
        module_description = "Not provided"
        program_aware = self.program_aware
        if program_aware:
            try:
                program_description = strip_prefix(
                    (
                        await self.describe_program(
                            program_code=self.program_code_string, program_example=task_demos, run=run
                        )
                    ).program_description
                )
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
                        run=run,
                    )
                ).module_description
            except Exception:
                logger.warning(
                    "Program-aware instruction proposal failed for this call; continuing without program context.",
                    exc_info=True,
                )
                program_aware = False
                program_description = "Not available"
                module_code = "Not provided"
                module_description = "Not provided"
        instruction_inputs = {
            "dataset_description": data_summary,
            "program_code": self.program_code_string,
            "module": module_code,
            "program_description": program_description,
            "module_description": module_description,
            "task_demos": task_demos,
            "tip": tip,
            "basic_instruction": basic_instruction,
            "previous_instructions": previous_instructions,
        }
        task_spec = get_task_spec(self.generate_module_instruction)
        filtered_inputs = {
            key: value
            for key, value in instruction_inputs.items()
            if key in task_spec.fields
            and task_spec.fields[key].role == "input"
            and (program_aware or key not in PROGRAM_AWARE_INPUT_KEYS)
        }
        instruct = await self.generate_module_instruction(**filtered_inputs, run=run)
        proposed_instruction = strip_prefix(instruct.proposed_instruction)
        return Prediction(proposed_instruction=proposed_instruction)


class GroundedProposer:
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
        self.prompt_model = prompt_model
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

    async def _ensure_data_summary(self, *, run) -> None:
        if self.data_summary is None and self.use_dataset_summary:
            self.data_summary = await create_dataset_summary(
                trainset=self._summary_trainset,
                view_data_batch_size=self._view_data_batch_size,
                prompt_model=resolve_optimizer_lm(self.prompt_model, run=run),
                run=run,
            )

    async def propose_instructions_for_program(
        self,
        trainset,
        program,
        demo_candidates,
        trial_logs: TrialLogs,
        num_candidates: int,
        *,
        run,
    ) -> dict[int, list[str]]:
        await self._ensure_data_summary(run=run)
        proposed_instructions = {}
        use_instruct_history = self.use_instruct_history
        if self.set_history_randomly:
            use_instruct_history = self.rng.random() < 0.5
        use_task_demos = self.use_task_demos and bool(demo_candidates)
        num_demos = num_candidates if not demo_candidates else max(len(demo_candidates[0]), 1)
        for pred_i, predictor in enumerate(program.predictors()):
            for demo_set_i in range(num_demos)[: min(num_candidates, num_demos)]:
                if pred_i not in proposed_instructions:
                    proposed_instructions[pred_i] = []
                selected_tip = None
                use_tip = self.use_tip
                if self.set_tip_randomly:
                    selected_tip_key = self.rng.choice(list(TIPS.keys()))
                    selected_tip = TIPS[selected_tip_key]
                    use_tip = bool(selected_tip)
                proposed_instructions[pred_i].append(
                    await self.propose_instruction_for_predictor(
                        program=program,
                        predictor=predictor,
                        pred_i=pred_i,
                        demo_candidates=demo_candidates,
                        demo_set_i=demo_set_i,
                        trial_logs=trial_logs,
                        tip=selected_tip,
                        use_task_demos=use_task_demos,
                        use_instruct_history=use_instruct_history,
                        use_tip=use_tip,
                        run=run,
                    )
                )
        return proposed_instructions

    async def propose_instruction_for_predictor(
        self,
        program,
        predictor,
        pred_i,
        demo_candidates,
        demo_set_i,
        trial_logs: TrialLogs,
        tip=None,
        *,
        use_task_demos: bool | None = None,
        use_instruct_history: bool | None = None,
        use_tip: bool | None = None,
        run,
    ) -> str:
        await self._ensure_data_summary(run=run)
        instruction_history = create_predictor_level_history_string(
            base_program=program, predictor_i=pred_i, trial_logs=trial_logs, top_n=MAX_INSTRUCT_IN_HISTORY
        )
        effective_use_task_demos = (
            self.use_task_demos and bool(demo_candidates) if use_task_demos is None else use_task_demos
        )
        base_use_instruct_history = self.use_instruct_history if use_instruct_history is None else use_instruct_history
        effective_use_instruct_history = base_use_instruct_history and bool(instruction_history)
        effective_use_tip = self.use_tip if use_tip is None else use_tip
        instruction_generator = GenerateModuleInstruction(
            program_code_string=self.program_code_string,
            use_dataset_summary=self.use_dataset_summary,
            program_aware=self.program_aware,
            use_task_demos=effective_use_task_demos,
            use_instruct_history=effective_use_instruct_history,
            use_tip=effective_use_tip,
            verbose=self.verbose,
        )
        rollout_lm = resolve_optimizer_lm(self.prompt_model, run=run).copy(temperature=self.init_temperature)
        with optimizer_lm_context(run, lm=rollout_lm, phase="propose.grounded", lm_role="prompt_model") as opt_run:
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
                    run=opt_run,
                )
            ).proposed_instruction
        if self.verbose:
            self.prompt_model.inspect_call_log(n=1)
        return strip_prefix(proposed_instruction)
