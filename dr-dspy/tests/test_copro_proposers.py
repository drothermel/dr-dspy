from __future__ import annotations

from typing import Any

import pytest

from dr_dspy import copro_proposers, dspy_runner
from dr_dspy.dspy_runner import PredictorRunResult
from dr_dspy.lm_utils import ModelConfig


class _Recorder:
    def __init__(self) -> None:
        self.lm_models: list[str] = []
        self.input_kwargs: list[dict[str, Any]] = []
        self.calls = 0


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    rec = _Recorder()

    def fake_build_logged_lm(*, model: str, **_kwargs: Any) -> object:
        rec.lm_models.append(model)
        return object()

    def fake_run_predictor(
        *, input_kwargs: dict[str, Any], **_kwargs: Any
    ) -> str:
        rec.input_kwargs.append(dict(input_kwargs))
        rec.calls += 1
        return f"  proposed #{rec.calls}  "

    def fake_predictor_run_result(
        text: str, _buffer: Any
    ) -> PredictorRunResult:
        return PredictorRunResult(
            text=text,
            usage_metadata={"total_tokens": 5},
            provider_cost=0.001,
        )

    monkeypatch.setattr(dspy_runner, "build_logged_lm", fake_build_logged_lm)
    monkeypatch.setattr(dspy_runner, "run_predictor", fake_run_predictor)
    monkeypatch.setattr(
        dspy_runner, "predictor_run_result", fake_predictor_run_result
    )
    return rec


def test_propose_basic_breadth_and_logging(recorder: _Recorder) -> None:
    proposals = copro_proposers.propose_basic(
        prompt_model=ModelConfig(model="prompt-model"),
        basic_instruction="seed instruction",
        breadth=3,
    )
    assert len(proposals) == 3
    assert proposals[0].instruction == "proposed #1"  # stripped
    assert proposals[0].usage == {"total_tokens": 5}
    assert proposals[0].cost == 0.001
    # prompt_model (not task_model) drives every proposal call.
    assert recorder.lm_models == ["prompt-model"] * 3
    assert recorder.input_kwargs[0] == {
        "basic_instruction": "seed instruction"
    }


def test_propose_given_attempts_formats_history(recorder: _Recorder) -> None:
    proposals = copro_proposers.propose_given_attempts(
        prompt_model=ModelConfig(model="prompt-model"),
        history=[("best instr", 0.9), ("worse instr", 0.2)],
        breadth=2,
    )
    assert len(proposals) == 2
    attempted = recorder.input_kwargs[0]["attempted_instructions"]
    assert "Instruction #1: best instr" in attempted
    assert "Score #1: 0.9000" in attempted
    assert "Instruction #2: worse instr" in attempted


def test_format_attempts_ordering() -> None:
    rendered = copro_proposers.format_attempts([("a", 1.0), ("b", 0.5)])
    assert rendered.splitlines()[0] == "Instruction #1: a"
    assert rendered.splitlines()[1] == "Score #1: 1.0000"
