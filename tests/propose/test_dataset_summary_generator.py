import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from dspy.primitives import Prediction
from dspy.propose.dataset_summary_generator import create_dataset_summary
from dspy.testing import DummyLM


def test_create_dataset_summary_empty_trainset_raises(make_run):
    run = make_run(lm=DummyLM([]))
    with pytest.raises(ValueError, match="trainset must be non-empty"):
        asyncio.run(
            create_dataset_summary(
                trainset=[],
                view_data_batch_size=2,
                prompt_model=DummyLM([]),
                run=run,
            )
        )


def test_create_dataset_summary_incremental_failure_propagates(make_run):
    trainset = [object() for _ in range(5)]
    prompt_model = DummyLM([{"observations": "initial"}, {"summary": "done"}])
    run = make_run(lm=prompt_model)
    call_count = 0

    async def predict_side_effect(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return Prediction(observations="initial")
        if call_count == 2:
            raise RuntimeError("incremental batch failed")
        return Prediction(summary="done")

    mock_predict = AsyncMock(side_effect=predict_side_effect)

    with patch("dspy.propose.dataset_summary_generator.Predict") as predict_cls:
        predict_cls.return_value = mock_predict
        with pytest.raises(RuntimeError, match="incremental batch failed"):
            asyncio.run(
                create_dataset_summary(
                    trainset=trainset,
                    view_data_batch_size=2,
                    prompt_model=prompt_model,
                    run=run,
                )
            )
