import pytest

from tests.adapters.assertions import assert_messages_exact
from tests.adapters.conftest import format_messages_and_lm_kwargs
from tests.adapters.golden.registry import ALL_GOLDEN_CASES


@pytest.mark.parametrize("case", ALL_GOLDEN_CASES, ids=lambda case: case.id)
def test_golden_prompt_format(case):
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=case.adapter_builder(),
        task_spec=case.scenario.task_spec,
        demos=list(case.scenario.demos),
        inputs=case.scenario.inputs,
        lm=case.scenario.lm,
        lm_kwargs=case.scenario.config,
    )
    if case.normalize is not None:
        messages = case.normalize(messages)
    assert_messages_exact(messages=messages, expected=case.messages)
    assert lm_kwargs == case.lm_kwargs
