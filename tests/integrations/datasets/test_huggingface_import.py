from dspy.integrations.datasets import huggingface as hf_module


def test_huggingface_loader_functions_exported() -> None:
    assert hf_module.examples_from_huggingface is not None
    assert hf_module.examples_from_csv is not None
    assert hf_module.examples_from_json is not None
    assert hf_module.examples_from_parquet is not None
