from types import SimpleNamespace

import pytest

from dspy.integrations.optimizers.optuna.distributions import get_param_distributions


def test_get_param_distributions_builds_instruction_and_demo_keys() -> None:
    pytest.importorskip("optuna")
    program = SimpleNamespace()
    program.predictors = lambda: [SimpleNamespace(), SimpleNamespace()]
    instruction_candidates = {0: ["a", "b"], 1: ["c"]}
    demo_candidates = {0: ["d1", "d2"], 1: ["d3"]}

    distributions = get_param_distributions(
        program=program,
        instruction_candidates=instruction_candidates,
        demo_candidates=demo_candidates,
    )

    assert set(distributions) == {
        "0_predictor_instruction",
        "1_predictor_instruction",
        "0_predictor_demos",
        "1_predictor_demos",
    }
