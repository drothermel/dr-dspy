from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from dspy.clients.finetune.service import FinetuneService
from dspy.clients.finetune.utils import TrainDataFormat
from dspy.clients.lm import LM
from dspy.errors import LMUnsupportedFeatureError
from dspy.integrations.finetune.openai import OpenAIProvider


def test_finetune_service_delegates_launch_and_kill():
    lm = LM("openai/gpt-4.1-mini")
    provider = MagicMock()
    service = FinetuneService(lm, finetune_provider=provider, launch_kwargs={"timeout": 30})

    service.launch()
    provider.launch.assert_called_once_with(lm, {"timeout": 30})

    service.kill()
    provider.kill.assert_called_once_with(lm, None)


def test_finetune_service_rejects_non_finetunable_provider():
    lm = LM("meta-llama/Llama-3.2-1B")
    service = FinetuneService(lm)

    with pytest.raises(LMUnsupportedFeatureError, match="does not support fine-tuning"):
        service.finetune(train_data=[{"messages": []}], train_data_format=TrainDataFormat.CHAT)


@patch.object(OpenAIProvider, "finetune", return_value="ft:new-model")
def test_finetune_service_returns_copied_lm_with_updated_model(mock_finetune):
    lm = LM("openai/gpt-4.1-mini")
    service = FinetuneService(lm, finetune_provider=OpenAIProvider())
    train_data = [{"messages": [{"role": "user", "content": "hi"}]}]

    job = service.finetune(train_data=train_data, train_data_format=TrainDataFormat.CHAT)
    result = job.result()

    assert isinstance(result, LM)
    assert result.model == "ft:new-model"
    assert result is not lm
