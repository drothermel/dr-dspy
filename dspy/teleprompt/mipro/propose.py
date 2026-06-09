import logging
from typing import TYPE_CHECKING, Any

from dspy.propose.grounded_proposer import GroundedProposer
from dspy.runtime.run_context import RunContext
from dspy.task_spec.predictor_context import get_task_spec
from dspy.teleprompt.mipro.settings import BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT

if TYPE_CHECKING:
    from dspy.teleprompt.mipro.optimizer import MIPROv2
logger = logging.getLogger(__name__)


async def propose_instructions(
    optimizer: "MIPROv2",
    program: Any,
    trainset: list,
    demo_candidates: dict[int, list] | None,
    view_data_batch_size: int,
    program_aware_proposer: bool,
    data_aware_proposer: bool,
    tip_aware_proposer: bool,
    fewshot_aware_proposer: bool,
    num_instruct_candidates: int,
    run: RunContext,
) -> dict[int, list[str]]:
    logger.info("\n==> STEP 2: PROPOSE INSTRUCTION CANDIDATES <==")
    logger.info(
        "We will use the few-shot examples from the previous step, a generated dataset summary, a summary of the program code, and a randomly selected prompting tip to propose instructions."
    )
    proposer = GroundedProposer(
        program=program,
        trainset=trainset,
        prompt_model=optimizer.prompt_model,
        view_data_batch_size=view_data_batch_size,
        program_aware=program_aware_proposer,
        use_dataset_summary=data_aware_proposer,
        use_task_demos=fewshot_aware_proposer,
        num_demos_in_context=BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT,
        use_tip=tip_aware_proposer,
        set_tip_randomly=tip_aware_proposer,
        use_instruct_history=False,
        set_history_randomly=False,
        verbose=optimizer.verbose,
        rng=optimizer.rng,
        init_temperature=optimizer.init_temperature,
    )
    logger.info(f"\nProposing N={num_instruct_candidates} instructions...\n")
    instruction_candidates = await proposer.propose_instructions_for_program(
        trainset=trainset,
        program=program,
        demo_candidates=demo_candidates,
        num_candidates=num_instruct_candidates,
        trial_logs={},
        run=run,
    )
    for i, pred in enumerate(program.predictors()):
        logger.info(f"Proposed Instructions for Predictor {i}:\n")
        instruction_candidates[i][0] = get_task_spec(pred).instructions
        for j, instruction in enumerate(instruction_candidates[i]):
            logger.info(f"{j}: {instruction}\n")
        logger.info("\n")
    return instruction_candidates
