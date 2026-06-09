import logging
import statistics
from collections import defaultdict

from pydantic import BaseModel

from dspy.core.types.config import LMConfig
from dspy.predict.predict import Predict
from dspy.primitives import Module
from dspy.runtime.run_context import RunContext
from dspy.teleprompt.compilation import CompileResult, CompileStats, ProgramCandidate
from dspy.teleprompt.compile_params import COPROCompileParams
from dspy.teleprompt.copro.task_specs import (
    BasicGenerateInstructionTaskSpec,
    GenerateInstructionGivenAttemptsTaskSpec,
)
from dspy.teleprompt.metrics import OptimizerMetric
from dspy.teleprompt.registry import register_teleprompter
from dspy.teleprompt.task_spec_context import get_task_spec, set_task_spec
from dspy.teleprompt.utils import make_optimizer_evaluator, optimizer_lm_context

logger = logging.getLogger(__name__)


@register_teleprompter(params=COPROCompileParams)
class COPRO:
    def __init__(
        self,
        prompt_model=None,
        metric: OptimizerMetric | None = None,
        breadth=10,
        depth=3,
        init_temperature=1.4,
        track_stats=False,
        **_kwargs,
    ) -> None:
        if breadth <= 1:
            raise ValueError("Breadth must be greater than 1")
        self.metric = metric
        self.breadth = breadth
        self.depth = depth
        self.init_temperature = init_temperature
        self.prompt_model = prompt_model
        self.track_stats = track_stats

    def _check_candidates_equal(self, candidate1, candidate2) -> bool:
        for p1, p2 in zip(candidate1["program"].predictors(), candidate2["program"].predictors(), strict=True):
            if get_task_spec(p1).instructions != get_task_spec(p2).instructions:
                return False
            *_, p1_last_field = get_task_spec(p1).fields.values()
            *_, p2_last_field = get_task_spec(p2).fields.values()
            if p1_last_field != p2_last_field:
                return False
        return True

    def _drop_duplicates(self, candidates):
        final_candidates = []
        last_batch = []
        last_batch_score = -1
        for c in candidates:
            repeat = False
            if c["score"] == last_batch_score:
                for c2 in last_batch:
                    if self._check_candidates_equal(candidate1=c, candidate2=c2):
                        repeat = True
                        break
                if not repeat:
                    last_batch.append(c)
            else:
                last_batch = [c]
                last_batch_score = c["score"]
            if not repeat:
                final_candidates.append(c)
        return final_candidates

    def _print_task_spec(self, predictor) -> None:
        task_spec = get_task_spec(predictor)
        logger.debug(f"i: {task_spec.instructions}")
        logger.debug(f"p: {list(task_spec.fields.values())[-1].prefix}")

    async def compile(self, student: Module, *, params: BaseModel, run: RunContext) -> CompileResult:
        params = COPROCompileParams.model_validate(params)
        module = student.deepcopy()
        evaluate_kwargs = params.evaluate.model_dump(exclude_none=True)
        trainset = params.trainset
        evaluate_call_kwargs = {
            key: value
            for key, value in evaluate_kwargs.items()
            if key
            in {
                "max_concurrency",
                "display_progress",
                "display_table",
                "save_as_csv",
                "save_as_json",
            }
        }
        extra_evaluate_kwargs = {key: value for key, value in evaluate_call_kwargs.items() if key != "max_concurrency"}
        evaluator = make_optimizer_evaluator(
            run,
            devset=trainset,
            metric=self.metric,
            max_concurrency=evaluate_kwargs.get("max_concurrency"),
            max_errors=evaluate_kwargs.get("max_errors"),
            **extra_evaluate_kwargs,
        )
        total_calls = 0
        results_best = {
            id(p): {"depth": [], "max": [], "average": [], "min": [], "std": []} for p in module.predictors()
        }
        results_latest = {
            id(p): {"depth": [], "max": [], "average": [], "min": [], "std": []} for p in module.predictors()
        }
        candidates = {}
        evaluated_candidates = defaultdict(dict)
        for predictor in module.predictors():
            basic_instruction = None
            basic_prefix = None
            *_, last_key = get_task_spec(predictor).fields.keys()
            basic_instruction = get_task_spec(predictor).instructions
            basic_prefix = get_task_spec(predictor).fields[last_key].prefix
            if self.prompt_model:
                with optimizer_lm_context(
                    run, lm=self.prompt_model, phase="copro.generate_instruction", lm_role="prompt_model"
                ) as opt_run:
                    instruct = await Predict(
                        BasicGenerateInstructionTaskSpec(),
                        config=LMConfig(n=self.breadth - 1, temperature=self.init_temperature),
                    )(basic_instruction=basic_instruction, run=opt_run)
            else:
                instruct = await Predict(
                    BasicGenerateInstructionTaskSpec(),
                    config=LMConfig(n=self.breadth - 1, temperature=self.init_temperature),
                )(basic_instruction=basic_instruction, run=run)
            instruct.completions.proposed_instruction.append(basic_instruction)
            instruct.completions.proposed_prefix_for_output_field.append(basic_prefix)
            candidates[id(predictor)] = instruct.completions
            evaluated_candidates[id(predictor)] = {}
        if self.prompt_model:
            logger.debug(f"{self.prompt_model.inspect_call_log(n=1)}")
        latest_candidates = candidates
        all_candidates = candidates
        module_clone = module.deepcopy()
        for d in range(self.depth):
            logger.info(f"Iteration Depth: {d + 1}/{self.depth}.")
            latest_scores = []
            for p_i, (p_old, p_new) in enumerate(zip(module.predictors(), module_clone.predictors(), strict=True)):
                candidates_ = latest_candidates[id(p_old)]
                if len(module.predictors()) > 1:
                    candidates_ = all_candidates[id(p_old)]
                for c_i, c in enumerate(candidates_):
                    instruction, prefix = (
                        c.proposed_instruction.strip('"').strip(),
                        c.proposed_prefix_for_output_field.strip('"').strip(),
                    )
                    *_, last_key = get_task_spec(p_new).fields.keys()
                    updated_task_spec = (
                        get_task_spec(p_new).with_instructions(instruction).with_updated_field(last_key, prefix=prefix)
                    )
                    set_task_spec(predictor=p_new, task_spec=updated_task_spec)
                    for i, predictor in enumerate(module_clone.predictors()):
                        logger.debug(f"Predictor {i + 1}")
                        self._print_task_spec(predictor)
                    logger.info(
                        f"At Depth {d + 1}/{self.depth}, Evaluating Prompt Candidate #{c_i + 1}/{len(candidates_)} for Predictor {p_i + 1} of {len(module.predictors())}."
                    )
                    score = (await evaluator(module_clone, run=run, devset=trainset, **evaluate_call_kwargs)).score
                    if self.prompt_model:
                        logger.debug(f"prompt_model.inspect_call_log(n=1) {self.prompt_model.inspect_call_log(n=1)}")
                    total_calls += 1
                    replace_entry = True
                    logger.debug(f"(instruction, prefix) {(instruction, prefix)}")
                    if (instruction, prefix) in evaluated_candidates[id(p_old)]:
                        if evaluated_candidates[id(p_old)][instruction, prefix]["score"] >= score:
                            replace_entry = False
                    if replace_entry:
                        evaluated_candidates[id(p_old)][instruction, prefix] = {
                            "score": score,
                            "program": module_clone.deepcopy(),
                            "instruction": instruction,
                            "prefix": prefix,
                            "depth": d,
                        }
                    if len(candidates_) - self.breadth <= c_i:
                        latest_scores.append(score)
                if self.track_stats:
                    results_latest[id(p_old)]["depth"].append(d)
                    results_latest[id(p_old)]["max"].append(max(latest_scores))
                    results_latest[id(p_old)]["average"].append(sum(latest_scores) / len(latest_scores))
                    results_latest[id(p_old)]["min"].append(min(latest_scores))
                    results_latest[id(p_old)]["std"].append(statistics.pstdev(latest_scores))
                best_candidate = max(evaluated_candidates[id(p_old)].values(), key=lambda candidate: candidate["score"])
                *_, last_key = get_task_spec(p_old).fields.keys()
                updated_task_spec = (
                    get_task_spec(p_new)
                    .with_instructions(best_candidate["instruction"])
                    .with_updated_field(last_key, prefix=best_candidate["prefix"])
                )
                set_task_spec(predictor=p_new, task_spec=updated_task_spec)
                logger.debug(
                    f"Updating Predictor {id(p_old)} to:\ni: {best_candidate['instruction']}\np: {best_candidate['prefix']}"
                )
                logger.debug("Full predictor with update: ")
                for i, predictor in enumerate(module_clone.predictors()):
                    logger.debug(f"Predictor {i}")
                    self._print_task_spec(predictor)
            if d == self.depth - 1:
                break
            new_candidates = {}
            for p_base in module.predictors():
                attempts = []
                shortest_len = self.breadth
                shortest_len = min(len(evaluated_candidates[id(p_base)]), shortest_len)
                best_predictors = list(evaluated_candidates[id(p_base)].values())
                best_predictors.sort(key=lambda x: x["score"], reverse=True)
                if self.track_stats:
                    scores = [x["score"] for x in best_predictors][:10]
                    results_best[id(p_base)]["depth"].append(d)
                    results_best[id(p_base)]["max"].append(max(scores))
                    results_best[id(p_base)]["average"].append(sum(scores) / len(scores))
                    results_best[id(p_base)]["min"].append(min(scores))
                    results_best[id(p_base)]["std"].append(statistics.pstdev(scores))
                for i in range(shortest_len - 1, -1, -1):
                    attempts.append(f"Instruction #{shortest_len - i}: {best_predictors[i]['instruction']}")
                    attempts.append(f"Prefix #{shortest_len - i}: {best_predictors[i]['prefix']}")
                    attempts.append(f"Resulting Score #{shortest_len - i}: {best_predictors[i]['score']}")
                if self.prompt_model:
                    with optimizer_lm_context(
                        run, lm=self.prompt_model, phase="copro.refine_instruction", lm_role="prompt_model"
                    ) as opt_run:
                        instr = await Predict(
                            GenerateInstructionGivenAttemptsTaskSpec(),
                            config=LMConfig(n=self.breadth, temperature=self.init_temperature),
                        )(attempted_instructions=attempts, run=opt_run)
                else:
                    instr = await Predict(
                        GenerateInstructionGivenAttemptsTaskSpec(),
                        config=LMConfig(n=self.breadth, temperature=self.init_temperature),
                    )(attempted_instructions=attempts, run=run)
                new_candidates[id(p_base)] = instr.completions
                all_candidates[id(p_base)].proposed_instruction.extend(instr.completions.proposed_instruction)
                all_candidates[id(p_base)].proposed_prefix_for_output_field.extend(
                    instr.completions.proposed_prefix_for_output_field
                )
            latest_candidates = new_candidates
        candidates = []
        for predictor in module.predictors():
            candidates.extend(list(evaluated_candidates[id(predictor)].values()))
            if self.track_stats:
                best_predictors = list(evaluated_candidates[id(predictor)].values())
                best_predictors.sort(key=lambda x: x["score"], reverse=True)
                scores = [x["score"] for x in best_predictors][:10]
                results_best[id(predictor)]["depth"].append(self.depth - 1)
                results_best[id(predictor)]["max"].append(max(scores))
                results_best[id(predictor)]["average"].append(sum(scores) / len(scores))
                results_best[id(predictor)]["min"].append(min(scores))
                results_best[id(predictor)]["std"].append(statistics.pstdev(scores))
        candidates.sort(key=lambda x: x["score"], reverse=True)
        candidates = self._drop_duplicates(candidates)
        best_program = candidates[0]["program"]
        program_candidates = [ProgramCandidate(score=entry["score"], program=entry["program"]) for entry in candidates]
        copro_depth_stats = None
        if self.track_stats:
            copro_depth_stats = {"results_best": results_best, "results_latest": results_latest}
        return CompileResult(
            program=best_program,
            candidates=program_candidates,
            stats=CompileStats(metric_calls=total_calls, copro_depth_stats=copro_depth_stats),
        )
