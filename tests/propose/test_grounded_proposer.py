import asyncio
import random
from typing import Any
from unittest.mock import AsyncMock

import pytest
from typing_extensions import override

from dspy.predict.predict import Predict
from dspy.primitives import Example, Prediction
from dspy.propose.grounded_proposer import (
    GenerateModuleInstruction,
    GroundedProposer,
    generate_instruction_class,
)
from dspy.teleprompt.task_spec_context import get_task_spec
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
            trainset=trainset,
            program=program,
            demo_candidates=demo_candidates,
            trial_logs={},
            num_candidates=1,
            run=run,
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


def test_generate_module_instruction_keeps_task_demos_for_demo_set_zero(make_run):
    program = Predict(ts("question -> answer"))
    demo_example = Example.from_record(
        {"question": "What is the capital of France?", "answer": "Paris", "augmented": True},
        input_keys=("question",),
    )
    demo_candidates = [[[demo_example]]]
    spec_predict = generate_instruction_class(
        use_dataset_summary=False,
        program_aware=False,
        use_task_demos=True,
    )
    mock_generate = AsyncMock(return_value=Prediction(proposed_instruction="instruction"))
    mock_generate.task_spec = get_task_spec(spec_predict)

    generator = GenerateModuleInstruction(
        program_code_string="code",
        use_dataset_summary=False,
        program_aware=False,
        use_task_demos=True,
    )
    generator.generate_module_instruction = mock_generate
    run = make_run(lm=DummyLM([{"proposed_instruction": "instruction"}]))
    asyncio.run(
        generator(
            demo_candidates=demo_candidates,
            pred_i=0,
            demo_set_i=0,
            program=program,
            previous_instructions="",
            data_summary="",
            run=run,
        )
    )
    assert mock_generate.await_args is not None
    captured = mock_generate.await_args.kwargs
    assert "France" in str(captured["task_demos"])
    assert captured["task_demos"] != "No task demos provided."


def test_generate_module_instruction_program_aware_failure_does_not_mutate_instance(make_run):
    program = Predict(ts("question -> answer"))
    generator = GenerateModuleInstruction(
        program_code_string="code",
        use_dataset_summary=False,
        program_aware=True,
        use_task_demos=False,
    )
    run = make_run(lm=DummyLM([{"proposed_instruction": "instruction"}]))

    async def failing_describe_module(**_kwargs):
        raise RuntimeError("describe failed")

    spec_predict = generate_instruction_class(
        use_dataset_summary=False,
        program_aware=True,
        use_task_demos=False,
    )
    mock_generate = AsyncMock(return_value=Prediction(proposed_instruction="instruction"))
    mock_generate.task_spec = get_task_spec(spec_predict)

    generator.describe_program = AsyncMock(return_value=Prediction(program_description="desc"))
    generator.describe_module = failing_describe_module
    generator.generate_module_instruction = mock_generate

    for _ in range(2):
        asyncio.run(
            generator(
                demo_candidates=[[[]]],
                pred_i=0,
                demo_set_i=0,
                program=program,
                previous_instructions="",
                data_summary="",
                run=run,
            )
        )
    assert generator.program_aware is True


def test_propose_instructions_preserves_instance_config_flags(make_run):
    prompt_model = DummyLM([{"proposed_instruction": "instruction"}] * 10)
    program = Predict(ts("question -> answer"))
    proposer = GroundedProposer(
        prompt_model=prompt_model,
        program=program,
        trainset=[],
        verbose=False,
        use_dataset_summary=False,
        program_aware=False,
        use_tip=True,
        use_instruct_history=True,
        set_tip_randomly=True,
        set_history_randomly=True,
        rng=random.Random(0),
    )
    run = make_run(lm=prompt_model)
    for _ in range(2):
        asyncio.run(
            proposer.propose_instructions_for_program(
                trainset=[],
                program=program,
                demo_candidates=None,
                trial_logs={},
                num_candidates=1,
                run=run,
            )
        )
    assert proposer.use_tip is True
    assert proposer.use_instruct_history is True
