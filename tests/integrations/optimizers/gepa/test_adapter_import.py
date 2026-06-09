import pytest


def test_gepa_adapter_exports_dspy_adapter() -> None:
    pytest.importorskip("gepa")

    import dspy.integrations.optimizers.gepa.adapter as adapter_module

    assert adapter_module.DspyAdapter is not None
    assert adapter_module.AsyncProposalFn is not None
