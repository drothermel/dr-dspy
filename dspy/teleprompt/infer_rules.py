import logging
import math
import random

from pydantic import BaseModel
from typing_extensions import override

from dspy.predict.chain_of_thought import ChainOfThought
from dspy.primitives.module import Module
from dspy.runtime.run_context import RunContext
from dspy.task_spec import input_field, make_task_spec, output_field
from dspy.teleprompt.bootstrap import BootstrapFewShot
from dspy.teleprompt.compile_params import BootstrapFewShotCompileParams, InferRulesCompileParams
from dspy.teleprompt.task_spec_context import get_task_spec, set_task_spec
from dspy.teleprompt.utils import make_optimizer_evaluator, optimizer_lm_context

logger = logging.getLogger(__name__)


class InferRules(BootstrapFewShot):
    def __init__(self, num_candidates=10, num_rules=10, max_concurrency=None, teacher_run=None, **kwargs) -> None:
        super().__init__(teacher_run=teacher_run, **kwargs)
        self.num_candidates = num_candidates
        self.num_rules = num_rules
        self.max_concurrency = max_concurrency
        self.rules_induction_program = RulesInductionProgram(num_rules, teacher_run=teacher_run)
        self.metric = kwargs.get("metric")
        self.max_errors = kwargs.get("max_errors")

    @override
    async def compile(self, student: Module, *, params: BaseModel, run: RunContext) -> Module:
        params = InferRulesCompileParams.model_validate(params)
        trainset = params.trainset
        valset = params.valset
        if valset is None:
            train_size = int(0.5 * len(trainset))
            trainset, valset = (trainset[:train_size], trainset[train_size:])
        await super().compile(
            student,
            params=BootstrapFewShotCompileParams(trainset=trainset, teacher=params.teacher),
            run=run,
        )
        original_program = self.student.deepcopy()
        all_predictors = [p for p in original_program.predictors() if hasattr(p, "task_spec")]
        instructions_list = [get_task_spec(p).instructions for p in all_predictors]
        best_score = -math.inf
        best_program = None
        for candidate_idx in range(self.num_candidates):
            candidate_program = original_program.deepcopy()
            candidate_predictors = [p for p in candidate_program.predictors() if hasattr(p, "task_spec")]
            for i, predictor in enumerate(candidate_predictors):
                set_task_spec(
                    predictor=predictor, task_spec=get_task_spec(predictor).with_instructions(instructions_list[i])
                )
            for i, predictor in enumerate(candidate_predictors):
                rules = await self.induce_natural_language_rules(predictor=predictor, trainset=trainset, run=run)
                set_task_spec(
                    predictor=predictor, task_spec=get_task_spec(predictor).with_instructions(instructions_list[i])
                )
                self.update_program_instructions(predictor=predictor, natural_language_rules=rules)
            score = await self.evaluate_program(program=candidate_program, dataset=valset, run=run)
            if score > best_score:
                best_score = score
                best_program = candidate_program
            logger.info(f"Evaluated Candidate {candidate_idx + 1} with score {score}. Current best score: {best_score}")
        logger.info(f"Final best score: {best_score}")
        return best_program if best_program is not None else original_program

    async def induce_natural_language_rules(self, predictor, trainset, *, run: RunContext):
        demos = self.get_predictor_demos(trainset=trainset, predictor=predictor)
        task_spec = get_task_spec(predictor)
        while True:
            examples_text = self.format_examples(demos=demos, task_spec=task_spec)
            try:
                return await self.rules_induction_program(examples_text, run=run)
            except Exception as e:
                assert (
                    isinstance(e, ValueError)
                    or e.__class__.__name__ == "BadRequestError"
                    or "ContextWindowExceededError" in str(e)
                )
                if len(demos) > 1:
                    demos = demos[:-1]
                else:
                    raise RuntimeError(
                        "Failed to generate natural language rules since a single example couldn't fit in the model's context window."
                    ) from e

    def update_program_instructions(self, predictor, natural_language_rules) -> None:
        task_spec = get_task_spec(predictor)
        set_task_spec(
            predictor=predictor,
            task_spec=task_spec.with_instructions(
                f"{task_spec.instructions}\n\nPlease adhere to the following rules when making your prediction:\n{natural_language_rules}"
            ),
        )

    def format_examples(self, demos, task_spec):
        examples_text = ""
        for demo in demos:
            input_fields = {k: v for k, v in demo.items() if k in task_spec.input_fields}
            output_fields = {k: v for k, v in demo.items() if k in task_spec.output_fields}
            input_text = "\n".join((f"{k}: {v}" for k, v in input_fields.items()))
            output_text = "\n".join((f"{k}: {v}" for k, v in output_fields.items()))
            examples_text += f"Input Fields:\n{input_text}\n\n=========\nOutput Fields:\n{output_text}\n\n"
        return examples_text

    def get_predictor_demos(self, trainset, predictor):
        task_spec = get_task_spec(predictor)
        return [
            {
                key: value
                for key, value in example.items()
                if key in task_spec.input_fields or key in task_spec.output_fields
            }
            for example in trainset
        ]

    async def evaluate_program(self, program, dataset, *, run: RunContext):
        evaluate = make_optimizer_evaluator(
            run,
            devset=dataset,
            metric=self.metric,
            max_concurrency=self.max_concurrency,
            max_errors=self.max_errors,
            display_table=False,
            display_progress=True,
        )
        return (await evaluate(program, run=run, metric=self.metric)).score


def _rules_induction_task_spec(num_rules):
    return make_task_spec(
        {
            "examples_text": input_field("examples_text", str, desc="Text containing examples"),
            "natural_language_rules": output_field(
                "natural_language_rules", str, desc="Induced natural language rules"
            ),
        },
        instructions=f"Given a set of examples, extract a list of {num_rules} concise and non-redundant natural language rules that provide clear guidance for performing the task. All rules should be actionable for a well-specified scope of examples of this general kind of task.",
        name="framework.infer_rules.induction",
    )


class RulesInductionProgram(Module):
    def __init__(self, num_rules, teacher_run: RunContext | None = None) -> None:
        super().__init__()
        self.rules_induction = ChainOfThought(_rules_induction_task_spec(num_rules))
        self.teacher_run = teacher_run
        self.rng = random.Random(0)

    async def aforward(self, examples_text, *, run: RunContext):
        teacher_run = (self.teacher_run or run).fork(optimization_trace=[])
        lm = teacher_run.lm.copy(temperature=1.0)
        with optimizer_lm_context(run, lm=lm, phase="infer_rules.induction", lm_role="teacher") as opt_run:
            rules = (await self.rules_induction(examples_text=examples_text, run=opt_run)).natural_language_rules
        return rules.strip()
