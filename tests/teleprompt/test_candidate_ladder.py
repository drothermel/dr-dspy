import asyncio
from typing import Any, cast
from unittest.mock import AsyncMock, Mock, patch

from dspy.predict.predict import Predict
from dspy.primitives import Example, Module
from dspy.teleprompt.candidate_ladder import (
    CandidateLadderConfig,
    CandidateSeedKind,
    RandomizedBootstrapSeed,
    generate_demo_candidate_sets,
    iter_candidate_seeds,
)
from dspy.testing import DummyLM
from tests.task_spec.helpers import ts


class DummyModule(Module):
    def __init__(self):
        super().__init__()

    async def _aforward_impl(self, *, run, options=None, **inputs):
        pass


def test_iter_candidate_seeds_includes_baselines_and_random():
    config = CandidateLadderConfig(
        num_random=2,
        max_labeled_demos=4,
        max_bootstrapped_demos=4,
    )
    seeds = iter_candidate_seeds(config)
    assert len(seeds) == 5
    assert seeds[0].kind == CandidateSeedKind.BASELINE
    assert seeds[1].kind == CandidateSeedKind.LABELED_FEWSHOT
    assert seeds[2].kind == CandidateSeedKind.BOOTSTRAP
    assert seeds[3] == RandomizedBootstrapSeed(index=0)
    assert seeds[4] == RandomizedBootstrapSeed(index=1)


def test_iter_candidate_seeds_respects_include_flags():
    config = CandidateLadderConfig(
        num_random=1,
        include_baseline=False,
        include_labeled_fewshot=False,
        include_bootstrap=False,
        max_labeled_demos=4,
        max_bootstrapped_demos=4,
    )
    seeds = iter_candidate_seeds(config)
    assert len(seeds) == 1
    assert seeds[0].kind == CandidateSeedKind.RANDOMIZED_BOOTSTRAP


def test_iter_candidate_seeds_skips_labeled_when_max_zero():
    config = CandidateLadderConfig(
        num_random=0,
        include_bootstrap=False,
        max_labeled_demos=0,
        max_bootstrapped_demos=4,
    )
    seeds = iter_candidate_seeds(config)
    assert [seed.kind for seed in seeds] == [CandidateSeedKind.BASELINE]


def test_generate_demo_candidate_sets_passes_metric_threshold_for_unshuffled(make_run):
    student = DummyModule()
    cast("Any", student).predictor = Predict(ts("input -> output"))
    trainset = [Example.from_record({"input": "test", "output": "test"}, input_keys=("input",))]
    lm = DummyLM([{"output": "test"}])
    run = make_run(lm=lm)
    config = CandidateLadderConfig(num_random=1, max_labeled_demos=1, max_bootstrapped_demos=1)
    with patch("dspy.teleprompt.candidate_ladder.BootstrapFewShot") as MockBootstrap:
        mock_instance = Mock()
        mock_instance.compile = AsyncMock(return_value=student)
        MockBootstrap.return_value = mock_instance
        asyncio.run(
            generate_demo_candidate_sets(
                student=student,
                config=config,
                trainset=trainset,
                metric=lambda _ex, _pred, _trace=None: 1.0,
                run=run,
                metric_threshold=0.9,
            )
        )
        calls = MockBootstrap.call_args_list
        assert len(calls) >= 1, "BootstrapFewShot was never called"
        for call in calls:
            _, kwargs = call
            assert "metric_threshold" in kwargs, f"metric_threshold missing from BootstrapFewShot call: {kwargs}"
            assert kwargs["metric_threshold"] == 0.9, f"metric_threshold={kwargs['metric_threshold']}, expected 0.9"
