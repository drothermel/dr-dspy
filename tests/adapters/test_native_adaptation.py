from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.types.citation import Citations
from dspy.adapters.types.reasoning import Reasoning
from dspy.clients.base_lm import BaseLM
from dspy.core.types import LMConfig, LMReasoningConfig, NativeAdaptationMode, ReasoningEffort
from dspy.task_spec import input_field, make_task_spec, output_field


class StubLM(BaseLM):
    def __init__(
        self,
        *,
        reasoning_adaptation_mode: NativeAdaptationMode = NativeAdaptationMode.ADAPT,
        citations_adaptation_mode: NativeAdaptationMode = NativeAdaptationMode.ADAPT,
        supports_reasoning: bool = True,
    ) -> None:
        super().__init__(model="stub/test")
        self._reasoning_adaptation_mode = reasoning_adaptation_mode
        self._citations_adaptation_mode = citations_adaptation_mode
        self._supports_reasoning = supports_reasoning

    @property
    def supports_reasoning(self) -> bool:
        return self._supports_reasoning

    @property
    def reasoning_adaptation_mode(self) -> NativeAdaptationMode:
        return self._reasoning_adaptation_mode

    @property
    def citations_adaptation_mode(self) -> NativeAdaptationMode:
        return self._citations_adaptation_mode


def test_reasoning_skip_mode_leaves_task_spec_unchanged():
    adapter = ChatAdapter()
    task_spec = make_task_spec(
        {
            "question": input_field("question", desc="q"),
            "reasoning": output_field("reasoning", Reasoning, desc="reasoning"),
            "answer": output_field("answer", desc="a"),
        },
        instructions="answer",
    )
    lm = StubLM(reasoning_adaptation_mode=NativeAdaptationMode.SKIP)
    config = LMConfig(reasoning=LMReasoningConfig(effort=ReasoningEffort.HIGH))
    result = adapter._adapt_reasoning_native(task_spec=task_spec, field_name="reasoning", lm=lm, config=config)
    assert "reasoning" in result.output_fields
    assert config.reasoning is not None


def test_reasoning_adapt_mode_removes_field_and_sets_config():
    adapter = ChatAdapter()
    task_spec = make_task_spec(
        {
            "question": input_field("question", desc="q"),
            "reasoning": output_field("reasoning", Reasoning, desc="reasoning"),
            "answer": output_field("answer", desc="a"),
        },
        instructions="answer",
    )
    lm = StubLM(reasoning_adaptation_mode=NativeAdaptationMode.ADAPT)
    config = LMConfig(reasoning=LMReasoningConfig(effort=ReasoningEffort.HIGH))
    result = adapter._adapt_reasoning_native(task_spec=task_spec, field_name="reasoning", lm=lm, config=config)
    assert "reasoning" not in result.output_fields
    assert config.reasoning is not None
    assert config.reasoning.effort == ReasoningEffort.HIGH


def test_citations_skip_mode_removes_field():
    adapter = ChatAdapter()
    task_spec = make_task_spec(
        {
            "question": input_field("question", desc="q"),
            "citations": output_field("citations", Citations, desc="cites"),
            "answer": output_field("answer", desc="a"),
        },
        instructions="answer",
    )
    lm = StubLM(citations_adaptation_mode=NativeAdaptationMode.SKIP)
    result = adapter._adapt_citations_native(task_spec=task_spec, field_name="citations", lm=lm)
    assert "citations" not in result.output_fields
