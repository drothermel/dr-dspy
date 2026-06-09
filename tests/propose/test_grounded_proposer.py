import asyncio
from typing import Any

import pytest
from typing_extensions import override

from dspy.predict.predict import Predict
from dspy.primitives import Example
from dspy.propose.grounded_proposer import GroundedProposer
from dspy.testing import DummyLM
from tests.task_spec.helpers import ts


@pytest.mark.parametrize(
    "demo_candidates",
    [None, [[[Example.from_record({"question": "What is the capital of France?", "answer": "Paris"})]]]],
)
def test_propose_instructions_for_program(demo_candidates, make_run):
    prompt_model = DummyLM([{"proposed_instruction": "instruction"}] * 10)
    program = Predict(ts("question -> answer"))
    trainset = []
    proposer = GroundedProposer(
        prompt_model=prompt_model,
        program=program,
        trainset=trainset,
        verbose=False,
        use_dataset_summary=False,
        program_aware=False,
    )
    run = make_run(lm=prompt_model)
    result = asyncio.run(
        proposer.propose_instructions_for_program(
            trainset=trainset, program=program, demo_candidates=demo_candidates, trial_logs={}, N=1, run=run
        )
    )
    assert isinstance(result, dict)
    assert len(result) == len(program.predictors())
    for pred_instructions in result.values():
        assert pred_instructions == ["instruction"]


@pytest.mark.parametrize(
    "demo_candidates",
    [None, [[[Example.from_record({"question": "What is the capital of France?", "answer": "Paris"})]]]],
)
def test_propose_instruction_for_predictor(demo_candidates, make_run):

    class TrackingDummyLM(DummyLM):
        @override
        def copy(self, **kwargs: Any):
            self.last_copy_kwargs = kwargs
            return super().copy(**kwargs)

    prompt_model = TrackingDummyLM([{"proposed_instruction": "instruction"}] * 10)
    program = Predict(ts("question -> answer"))
    proposer = GroundedProposer(
        prompt_model=prompt_model,
        program=program,
        trainset=[],
        verbose=False,
        use_dataset_summary=False,
        program_aware=False,
        use_tip=False,
        init_temperature=0.7,
    )
    run = make_run(lm=prompt_model)
    result = asyncio.run(
        proposer.propose_instruction_for_predictor(
            program=program,
            predictor=None,
            pred_i=0,
            demo_candidates=demo_candidates,
            demo_set_i=0,
            trial_logs={},
            tip=None,
            run=run,
        )
    )
    assert result == "instruction"
    assert prompt_model.last_copy_kwargs["temperature"] == 0.7
