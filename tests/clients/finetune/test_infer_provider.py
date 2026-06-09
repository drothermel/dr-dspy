import pytest

from dspy.clients.finetune import lm as finetune_lm
from dspy.clients.finetune.provider import DefaultFinetuneProvider
from dspy.clients.lm import LM
from dspy.integrations.finetune.databricks import DatabricksProvider
from dspy.integrations.finetune.local import LocalProvider
from dspy.integrations.finetune.openai import OpenAIProvider


@pytest.mark.parametrize(
    ("model", "expected_type"),
    [
        ("databricks/my_endpoint", DatabricksProvider),
        ("local:/tmp/out", LocalProvider),
        ("openai/gpt-4.1-mini", OpenAIProvider),
        ("ft:abc123", OpenAIProvider),
        ("meta-llama/Llama-3.2-1B", DefaultFinetuneProvider),
    ],
)
def test_infer_provider(model, expected_type):
    provider = finetune_lm.infer_provider(model)
    assert isinstance(provider, expected_type)


def test_lm_infers_databricks_provider_without_explicit_provider():
    lm = LM(model="databricks/foo")
    assert isinstance(lm.provider, DatabricksProvider)
    assert lm.provider.finetunable is True
