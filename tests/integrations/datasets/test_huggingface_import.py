from dspy.integrations.datasets import huggingface as hf_module


def test_huggingface_dataloader_exports() -> None:
    assert hf_module.HuggingFaceDataLoader is not None
