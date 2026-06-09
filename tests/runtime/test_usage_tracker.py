import asyncio
from typing import Any, cast
from unittest import mock

from pydantic import BaseModel

from dspy.adapters.json_adapter import JSONAdapter
from dspy.clients.lm import LM
from dspy.predict.chain_of_thought import ChainOfThought
from dspy.predict.parallel import Parallel
from dspy.runtime import TelemetryConfig
from dspy.utils.usage_tracker import UsageTracker, track_usage
from tests.task_spec.helpers import ts


def test_add_usage_entry():
    tracker = UsageTracker()
    usage_entry = {
        "prompt_tokens": 1117,
        "completion_tokens": 46,
        "total_tokens": 1163,
        "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
        "completion_tokens_details": {
            "reasoning_tokens": 0,
            "audio_tokens": 0,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 0,
        },
    }
    tracker.add_usage("gpt-4o-mini", usage_entry)
    assert len(tracker.usage_data["gpt-4o-mini"]) == 1
    assert tracker.usage_data["gpt-4o-mini"][0] == usage_entry


def test_get_total_tokens():
    tracker = UsageTracker()
    usage_entries = [
        {
            "prompt_tokens": 1117,
            "completion_tokens": 46,
            "total_tokens": 1163,
            "prompt_tokens_details": {"cached_tokens": 200, "audio_tokens": 50},
            "completion_tokens_details": {
                "reasoning_tokens": 20,
                "audio_tokens": 10,
                "accepted_prediction_tokens": 16,
                "rejected_prediction_tokens": 0,
            },
        },
        {
            "prompt_tokens": 800,
            "completion_tokens": 100,
            "total_tokens": 900,
            "prompt_tokens_details": {"cached_tokens": 300, "audio_tokens": 0},
            "completion_tokens_details": {
                "reasoning_tokens": 50,
                "audio_tokens": 0,
                "accepted_prediction_tokens": 40,
                "rejected_prediction_tokens": 10,
            },
        },
        {
            "prompt_tokens": 500,
            "completion_tokens": 80,
            "total_tokens": 580,
            "prompt_tokens_details": {"cached_tokens": 100, "audio_tokens": 25},
            "completion_tokens_details": {
                "reasoning_tokens": 30,
                "audio_tokens": 15,
                "accepted_prediction_tokens": 25,
                "rejected_prediction_tokens": 10,
            },
        },
    ]
    for entry in usage_entries:
        tracker.add_usage("gpt-4o-mini", entry)
    total_usage = tracker.get_total_tokens()
    assert "gpt-4o-mini" in total_usage
    assert total_usage["gpt-4o-mini"]["prompt_tokens"] == 2417
    assert total_usage["gpt-4o-mini"]["completion_tokens"] == 226
    assert total_usage["gpt-4o-mini"]["total_tokens"] == 2643
    assert total_usage["gpt-4o-mini"]["prompt_tokens_details"]["cached_tokens"] == 600
    assert total_usage["gpt-4o-mini"]["prompt_tokens_details"]["audio_tokens"] == 75
    assert total_usage["gpt-4o-mini"]["completion_tokens_details"]["reasoning_tokens"] == 100
    assert total_usage["gpt-4o-mini"]["completion_tokens_details"]["audio_tokens"] == 25
    assert total_usage["gpt-4o-mini"]["completion_tokens_details"]["accepted_prediction_tokens"] == 81
    assert total_usage["gpt-4o-mini"]["completion_tokens_details"]["rejected_prediction_tokens"] == 20


def test_track_usage_with_multiple_models(make_run):
    tracker = UsageTracker()
    usage_entries = [
        {
            "model": "gpt-4o-mini",
            "usage": {
                "prompt_tokens": 1117,
                "completion_tokens": 46,
                "total_tokens": 1163,
                "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
                "completion_tokens_details": {
                    "reasoning_tokens": 0,
                    "audio_tokens": 0,
                    "accepted_prediction_tokens": 0,
                    "rejected_prediction_tokens": 0,
                },
            },
        },
        {
            "model": "gpt-3.5-turbo",
            "usage": {
                "prompt_tokens": 800,
                "completion_tokens": 100,
                "total_tokens": 900,
                "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
                "completion_tokens_details": {
                    "reasoning_tokens": 0,
                    "audio_tokens": 0,
                    "accepted_prediction_tokens": 0,
                    "rejected_prediction_tokens": 0,
                },
            },
        },
    ]
    for entry in usage_entries:
        tracker.add_usage(cast("str", entry["model"]), cast("dict[str, Any]", entry["usage"]))
    total_usage = tracker.get_total_tokens()
    assert "gpt-4o-mini" in total_usage
    assert "gpt-3.5-turbo" in total_usage
    assert total_usage["gpt-4o-mini"]["total_tokens"] == 1163
    assert total_usage["gpt-3.5-turbo"]["total_tokens"] == 900


def test_track_usage_context_manager(lm_for_test, make_run):
    lm = LM(lm_for_test, temperature=0.0)
    run = make_run(lm=lm)
    predict = ChainOfThought(ts("question -> answer"))
    with track_usage(run) as tracker:
        asyncio.run(predict(question="What is the capital of France?", run=run))
        asyncio.run(predict(question="What is the capital of Italy?", run=run))
    assert len(tracker.usage_data) > 0
    assert len(tracker.usage_data[lm_for_test]) == 2
    total_usage = tracker.get_total_tokens()
    assert lm_for_test in total_usage
    assert len(total_usage.keys()) == 1
    assert isinstance(total_usage[lm_for_test], dict)


def test_merge_usage_entries_with_new_keys():
    tracker = UsageTracker()
    tracker.add_usage("model-x", {"prompt_tokens": 5})
    tracker.add_usage("model-x", {"completion_tokens": 2})
    total_usage = tracker.get_total_tokens()
    assert total_usage["model-x"]["prompt_tokens"] == 5
    assert total_usage["model-x"]["completion_tokens"] == 2


def test_merge_usage_entries_with_none_values():
    tracker = UsageTracker()
    usage_entries = [
        {
            "model": "gpt-4o-mini",
            "usage": {
                "prompt_tokens": 1117,
                "completion_tokens": 46,
                "total_tokens": 1163,
                "prompt_tokens_details": None,
                "completion_tokens_details": {},
            },
        },
        {
            "model": "gpt-4o-mini",
            "usage": {
                "prompt_tokens": 800,
                "completion_tokens": 100,
                "total_tokens": 900,
                "prompt_tokens_details": None,
                "completion_tokens_details": None,
            },
        },
        {
            "model": "gpt-4o-mini",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 100,
                "total_tokens": 200,
                "prompt_tokens_details": {"cached_tokens": 50, "audio_tokens": 50},
                "completion_tokens_details": None,
            },
        },
        {
            "model": "gpt-4o-mini",
            "usage": {
                "prompt_tokens": 800,
                "completion_tokens": 100,
                "total_tokens": 900,
                "prompt_tokens_details": None,
                "completion_tokens_details": {
                    "reasoning_tokens": 1,
                    "audio_tokens": 1,
                    "accepted_prediction_tokens": 1,
                    "rejected_prediction_tokens": 1,
                },
            },
        },
    ]
    for entry in usage_entries:
        tracker.add_usage(cast("str", entry["model"]), cast("dict[str, Any]", entry["usage"]))
    total_usage = tracker.get_total_tokens()
    assert total_usage["gpt-4o-mini"]["prompt_tokens"] == 2817
    assert total_usage["gpt-4o-mini"]["completion_tokens"] == 346
    assert total_usage["gpt-4o-mini"]["total_tokens"] == 3163
    assert total_usage["gpt-4o-mini"]["prompt_tokens_details"]["cached_tokens"] == 50
    assert total_usage["gpt-4o-mini"]["prompt_tokens_details"]["audio_tokens"] == 50
    assert total_usage["gpt-4o-mini"]["completion_tokens_details"]["reasoning_tokens"] == 1
    assert total_usage["gpt-4o-mini"]["completion_tokens_details"]["audio_tokens"] == 1
    assert total_usage["gpt-4o-mini"]["completion_tokens_details"]["accepted_prediction_tokens"] == 1
    assert total_usage["gpt-4o-mini"]["completion_tokens_details"]["rejected_prediction_tokens"] == 1


def test_merge_usage_entries_with_pydantic_models():
    tracker = UsageTracker()

    class CacheCreationTokenDetails(BaseModel):
        ephemeral_5m_input_tokens: int
        ephemeral_1h_input_tokens: int

    class PromptTokensDetailsWrapper(BaseModel):
        audio_tokens: int | None
        cached_tokens: int
        text_tokens: int | None
        image_tokens: int | None
        cache_creation_tokens: int
        cache_creation_token_details: CacheCreationTokenDetails

    usage_entries = [
        {
            "model": "gpt-4o-mini",
            "usage": {
                "prompt_tokens": 1117,
                "completion_tokens": 46,
                "total_tokens": 1163,
                "prompt_tokens_details": PromptTokensDetailsWrapper(
                    audio_tokens=None,
                    cached_tokens=3,
                    text_tokens=None,
                    image_tokens=None,
                    cache_creation_tokens=0,
                    cache_creation_token_details=CacheCreationTokenDetails(
                        ephemeral_5m_input_tokens=5, ephemeral_1h_input_tokens=0
                    ),
                ),
                "completion_tokens_details": {},
            },
        },
        {
            "model": "gpt-4o-mini",
            "usage": {
                "prompt_tokens": 800,
                "completion_tokens": 100,
                "total_tokens": 900,
                "prompt_tokens_details": PromptTokensDetailsWrapper(
                    audio_tokens=None,
                    cached_tokens=3,
                    text_tokens=None,
                    image_tokens=None,
                    cache_creation_tokens=0,
                    cache_creation_token_details=CacheCreationTokenDetails(
                        ephemeral_5m_input_tokens=5, ephemeral_1h_input_tokens=0
                    ),
                ),
                "completion_tokens_details": None,
            },
        },
        {
            "model": "gpt-4o-mini",
            "usage": {
                "prompt_tokens": 800,
                "completion_tokens": 100,
                "total_tokens": 900,
                "prompt_tokens_details": PromptTokensDetailsWrapper(
                    audio_tokens=None,
                    cached_tokens=3,
                    text_tokens=None,
                    image_tokens=None,
                    cache_creation_tokens=0,
                    cache_creation_token_details=CacheCreationTokenDetails(
                        ephemeral_5m_input_tokens=5, ephemeral_1h_input_tokens=0
                    ),
                ),
                "completion_tokens_details": {
                    "reasoning_tokens": 1,
                    "audio_tokens": 1,
                    "accepted_prediction_tokens": 1,
                    "rejected_prediction_tokens": 1,
                },
            },
        },
    ]
    for entry in usage_entries:
        tracker.add_usage(cast("str", entry["model"]), cast("dict[str, Any]", entry["usage"]))
    total_usage = tracker.get_total_tokens()
    assert total_usage["gpt-4o-mini"]["prompt_tokens"] == 2717
    assert total_usage["gpt-4o-mini"]["completion_tokens"] == 246
    assert total_usage["gpt-4o-mini"]["total_tokens"] == 2963
    assert total_usage["gpt-4o-mini"]["prompt_tokens_details"]["cached_tokens"] == 9
    assert (
        total_usage["gpt-4o-mini"]["prompt_tokens_details"]["cache_creation_token_details"]["ephemeral_5m_input_tokens"]
        == 15
    )
    assert total_usage["gpt-4o-mini"]["completion_tokens_details"]["reasoning_tokens"] == 1
    assert total_usage["gpt-4o-mini"]["completion_tokens_details"]["audio_tokens"] == 1
    assert total_usage["gpt-4o-mini"]["completion_tokens_details"]["accepted_prediction_tokens"] == 1
    assert total_usage["gpt-4o-mini"]["completion_tokens_details"]["rejected_prediction_tokens"] == 1


def test_parallel_executor_with_usage_tracker(make_run):
    parent_tracker = UsageTracker()
    mock_lm = mock.MagicMock(spec=LM)
    mock_lm.return_value = ['{"answer": "Mocked answer"}']
    mock_lm.kwargs = {}
    mock_lm.model = "openai/gpt-4o-mini"
    run = make_run(lm=mock_lm, adapter=JSONAdapter())

    async def task1(**_kwargs: object):
        worker_run = run.fork(usage_tracker=UsageTracker())
        worker_run.usage_tracker.add_usage(
            "openai/gpt-4o-mini", {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}
        )
        return worker_run.usage_tracker.get_total_tokens()

    async def task2(**_kwargs: object):
        worker_run = run.fork(usage_tracker=UsageTracker())
        worker_run.usage_tracker.add_usage(
            "openai/gpt-4o-mini", {"prompt_tokens": 80, "completion_tokens": 15, "total_tokens": 95}
        )
        return worker_run.usage_tracker.get_total_tokens()

    run = run.fork(usage_tracker=parent_tracker, telemetry=TelemetryConfig(track_usage=True))
    executor = Parallel()
    results = asyncio.run(executor([(task1, {}), (task2, {})], run=run))
    usage1 = cast("dict[str, dict[str, Any]]", results[0])
    usage2 = cast("dict[str, dict[str, Any]]", results[1])
    assert usage1["openai/gpt-4o-mini"]["prompt_tokens"] == 50
    assert usage1["openai/gpt-4o-mini"]["completion_tokens"] == 10
    assert usage2["openai/gpt-4o-mini"]["prompt_tokens"] == 80
    assert usage2["openai/gpt-4o-mini"]["completion_tokens"] == 15
    assert len(parent_tracker.usage_data) == 0
