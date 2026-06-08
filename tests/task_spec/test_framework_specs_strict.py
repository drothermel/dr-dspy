import pytest

from dspy.adapters.base import ToolCallResultsTaskSpec
from dspy.evaluate.auto_evaluation import (
    AnswerCompletenessTaskSpec,
    AnswerGroundednessTaskSpec,
    DecompositionalSemanticRecallPrecisionTaskSpec,
    SemanticRecallPrecisionTaskSpec,
)
from dspy.predict.chain_of_thought import ChainOfThought
from dspy.predict.refine import OfferFeedbackTaskSpec
from dspy.predict.rlm import FrameworkRlmSubQueryTaskSpec
from dspy.task_spec import TaskSpec, input_field, output_field
from dspy.teleprompt.avatar_optimizer import ComparatorTaskSpec, FeedbackBasedInstructionTaskSpec
from dspy.teleprompt.copro_optimizer import BasicGenerateInstructionTaskSpec, GenerateInstructionGivenAttemptsTaskSpec
from dspy.teleprompt.gepa.instruction_proposal import GenerateEnhancedMultimodalInstructionTaskSpec
from dspy.teleprompt.gepa.task_specs import FrameworkGepaInstructionProposalTaskSpec
from dspy.teleprompt.simba_utils import SimbaOfferFeedbackTaskSpec
from dspy.utils.transparency import collect_task_spec_violations, is_placeholder_desc

FRAMEWORK_SPECS: list[type[TaskSpec]] = [
    ToolCallResultsTaskSpec,
    SemanticRecallPrecisionTaskSpec,
    DecompositionalSemanticRecallPrecisionTaskSpec,
    AnswerCompletenessTaskSpec,
    AnswerGroundednessTaskSpec,
    BasicGenerateInstructionTaskSpec,
    GenerateInstructionGivenAttemptsTaskSpec,
    SimbaOfferFeedbackTaskSpec,
    ComparatorTaskSpec,
    FeedbackBasedInstructionTaskSpec,
    FrameworkGepaInstructionProposalTaskSpec,
    GenerateEnhancedMultimodalInstructionTaskSpec,
    OfferFeedbackTaskSpec,
    FrameworkRlmSubQueryTaskSpec,
]


@pytest.mark.parametrize("spec_cls", FRAMEWORK_SPECS)
def test_framework_task_specs_have_explicit_field_descs(spec_cls):
    spec = spec_cls()
    violations = collect_task_spec_violations(spec)
    assert violations == [], f"{spec_cls.__name__} has placeholder descs: {violations}"
    assert spec.name.startswith("framework.")


def test_chain_of_thought_reasoning_field_has_explicit_desc():
    class QATaskSpec(TaskSpec):
        name: str = "QA"
        instructions: str = "Answer."
        inputs: tuple = (input_field("question", desc="Question."),)
        outputs: tuple = (output_field("answer", desc="Answer."),)

    cot = ChainOfThought(QATaskSpec())
    reasoning_field = cot.predict.task_spec.output_fields["reasoning"]
    assert not is_placeholder_desc(reasoning_field.desc, "reasoning")
