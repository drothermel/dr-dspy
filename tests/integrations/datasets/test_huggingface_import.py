import pytest


def test_huggingface_dataloader_exports() -> None:
    pytest.importorskip("datasets")

    from dspy.integrations.datasets import huggingface as hf_module

    assert hf_module.HuggingFaceDataLoader is not None
