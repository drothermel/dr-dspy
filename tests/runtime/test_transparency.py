import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from dspy.adapters.json_adapter import JSONAdapter
from dspy.core.types import LMConfig
from dspy.predict.predict import Predict
from dspy.runtime import CallLogMode, TelemetryConfig, TransparencyMode
from dspy.runtime.transparency.report import enforce_compiled_call_transparency
from dspy.runtime.transparency.resolve import LM_CONFIG_PROVENANCE_FIELDS
from dspy.runtime.transparency.types import CompiledCall, TransparencyViolation
from dspy.runtime.transparency.validate import collect_compiled_call_violations
from dspy.task_spec import TaskSpec, input_field, output_field
from dspy.testing import DummyLM


class SampleTaskSpec(TaskSpec):
    name: str = "Sample"
    instructions: str = "Do the thing."
    inputs: tuple = (input_field("question", desc="The user question."),)
    outputs: tuple = (output_field("answer", desc="The answer."),)


def test_provenance_fields_match_lm_config():
    expected = set(LMConfig.model_fields) - {"extensions"}
    assert set(LM_CONFIG_PROVENANCE_FIELDS) == expected


def test_collect_compiled_call_violations_is_pure_no_raise_or_log(make_run):
    call = CompiledCall(call_id="1", adapter_class="", lm_model="openai/gpt-4o-mini", cache=False)
    with (
        patch.object(logging.getLogger("dspy.runtime.transparency.report"), "warning") as warn_mock,
        patch.object(logging.getLogger("dspy.runtime.transparency.report"), "info") as info_mock,
    ):
        violations = collect_compiled_call_violations(call)
    assert violations
    warn_mock.assert_not_called()
    info_mock.assert_not_called()


def test_enforce_compiled_call_transparency_strict_raises_under_strict_mode(make_run):
    call = CompiledCall(call_id="1", adapter_class="", lm_model="openai/gpt-4o-mini", cache=False)
    with pytest.raises(TransparencyViolation, match="adapter not configured"):
        enforce_compiled_call_transparency(call, TransparencyMode.strict)


def test_enforce_compiled_call_transparency_warn_reports_config_violations(make_run):
    call = CompiledCall(
        call_id="1",
        adapter_class="JSONAdapter",
        original_task_spec=SampleTaskSpec(),
        processed_task_spec=SampleTaskSpec(),
        config=LMConfig(temperature=0.0, max_tokens=100),
        lm_model="openai/gpt-4o-mini",
        cache=False,
    )
    violations = enforce_compiled_call_transparency(call, TransparencyMode.warn)
    assert violations == []


def test_enforce_compiled_call_transparency_strict_raises_on_missing_adapter(make_run):
    call = CompiledCall(call_id="1", adapter_class="", lm_model="openai/gpt-4o-mini", cache=False)
    with pytest.raises(TransparencyViolation, match="adapter not configured"):
        enforce_compiled_call_transparency(call, TransparencyMode.strict)


def test_enforce_compiled_call_transparency_warn_mode_does_not_raise(make_run):
    call = CompiledCall(call_id="1", adapter_class="", lm_model="openai/gpt-4o-mini", cache=False)
    violations = enforce_compiled_call_transparency(call, TransparencyMode.warn)
    assert violations


def test_enforce_compiled_call_transparency_strict_reports_missing_max_tokens_from_lm_kwargs(make_run):
    call = CompiledCall(
        call_id="1",
        adapter_class="JSONAdapter",
        original_task_spec=SampleTaskSpec(),
        processed_task_spec=SampleTaskSpec(),
        config=LMConfig(temperature=0.0),
        lm_model="openai/gpt-4o-mini",
        lm_kwargs={},
        cache=False,
    )
    with pytest.raises(TransparencyViolation, match="max_tokens"):
        enforce_compiled_call_transparency(call, TransparencyMode.strict)


def test_enforce_compiled_call_transparency_strict_passes_when_max_tokens_in_lm_kwargs(make_run):
    call = CompiledCall(
        call_id="1",
        adapter_class="JSONAdapter",
        original_task_spec=SampleTaskSpec(),
        processed_task_spec=SampleTaskSpec(),
        config=LMConfig(temperature=0.0),
        lm_model="openai/gpt-4o-mini",
        lm_kwargs={"max_tokens": 4000},
        cache=False,
    )
    enforce_compiled_call_transparency(call, TransparencyMode.strict)


def test_enforce_compiled_call_transparency_strict_passes_for_explicit_call(make_run):
    call = CompiledCall(
        call_id="1",
        adapter_class="JSONAdapter",
        original_task_spec=SampleTaskSpec(),
        processed_task_spec=SampleTaskSpec(),
        config=LMConfig(temperature=0.0, max_tokens=100),
        lm_model="openai/gpt-4o-mini",
        cache=False,
    )
    enforce_compiled_call_transparency(call, TransparencyMode.strict)


def test_predict_lm_call_appends_jsonl(tmp_path, monkeypatch, make_run):
    import asyncio

    monkeypatch.setenv("DSPY_LOG_DIR", str(tmp_path))
    json_adapter = JSONAdapter()
    lm = DummyLM([{"answer": "42"}], adapter=json_adapter)
    run = make_run(
        lm=lm,
        adapter=json_adapter,
        telemetry=TelemetryConfig(transparency=TransparencyMode.strict, call_log=CallLogMode.both),
    )
    predict = Predict(SampleTaskSpec())
    asyncio.run(predict(question="2+2", run=run))
    calls_files = list(Path(tmp_path).rglob("calls.jsonl"))
    assert len(calls_files) == 1
    record = json.loads(calls_files[0].read_text(encoding="utf-8").strip())
    assert record["caller"]["phase"] == "predict"
    assert record["lm"]["model"] == "dummy"
