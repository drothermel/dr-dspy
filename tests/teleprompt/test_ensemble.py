import asyncio
from typing import cast

import pytest

from dspy.primitives import Module
from dspy.teleprompt.compile_params import EnsembleCompileParams
from dspy.teleprompt.ensemble import Ensemble
from dspy.testing import DummyLM


class MockProgram(Module):
    def __init__(self, output):
        super().__init__()
        self.output = output

    async def _aforward_impl(self, *args: object, **kwargs: object):
        return self.output


def mock_reduce_fn(outputs):
    return sum(outputs) / len(outputs)


def test_ensemble_without_reduction(make_run):
    run = make_run(lm=DummyLM([{}]))
    programs = [MockProgram(i) for i in range(5)]
    ensemble = Ensemble()
    result = asyncio.run(
        ensemble.compile(Module(), params=EnsembleCompileParams(programs=cast("list[Module]", programs)), run=run)
    )
    ensembled_program = result.program
    outputs = asyncio.run(ensembled_program(run=run))
    assert len(outputs) == 5, "Ensemble did not combine the correct number of outputs"


def test_ensemble_with_reduction(make_run):
    run = make_run(lm=DummyLM([{}]))
    programs = [MockProgram(i) for i in range(5)]
    ensemble = Ensemble(reduce_fn=mock_reduce_fn)
    result = asyncio.run(
        ensemble.compile(Module(), params=EnsembleCompileParams(programs=cast("list[Module]", programs)), run=run)
    )
    ensembled_program = result.program
    output = asyncio.run(ensembled_program(run=run))
    expected_output = sum(range(5)) / 5
    assert output == expected_output, "Ensemble did not correctly apply the reduce_fn"


def test_ensemble_with_size_limitation(make_run):
    run = make_run(lm=DummyLM([{}]))
    programs = [MockProgram(i) for i in range(10)]
    ensemble_size = 3
    ensemble = Ensemble(size=ensemble_size)
    result = asyncio.run(
        ensemble.compile(Module(), params=EnsembleCompileParams(programs=cast("list[Module]", programs)), run=run)
    )
    ensembled_program = result.program
    outputs = asyncio.run(ensembled_program(run=run))
    assert len(outputs) == ensemble_size, "Ensemble did not respect the specified size limitation"


def test_ensemble_deterministic_behavior():
    with pytest.raises(AssertionError, match=r"Deterministic ensemble is not supported"):
        Ensemble(deterministic=True)
