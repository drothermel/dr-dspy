import pytest

from dspy.adapters.base import ToolCallResultsTaskSpec
from dspy.adapters.two_step.task_specs import build_extractor_task_spec
from dspy.integrations.optimizers.gepa.task_specs import (
    FrameworkGepaInstructionProposalTaskSpec,
    GenerateEnhancedMultimodalInstructionTaskSpec,
)
from dspy.predict.chain_of_thought import ChainOfThought
from dspy.propose.task_specs import (
    DatasetDescriptorTaskSpec,
    DatasetDescriptorWithPriorObservationsTaskSpec,
    DescribeModuleTaskSpec,
    DescribeProgramTaskSpec,
    ObservationSummarizerTaskSpec,
    generate_instruction_task_spec,
)
from dspy.task_spec import TaskSpec, input_field, make_task_spec, output_field
from dspy.task_spec.framework.evaluate import (
    AnswerCompletenessTaskSpec,
    AnswerGroundednessTaskSpec,
    DecompositionalSemanticRecallPrecisionTaskSpec,
    SemanticRecallPrecisionTaskSpec,
)
from dspy.task_spec.framework.refine import OfferFeedbackTaskSpec
from dspy.task_spec.framework.rlm import FrameworkRlmSubQueryTaskSpec
from dspy.teleprompt.avatar.task_specs import ComparatorTaskSpec, FeedbackBasedInstructionTaskSpec
from dspy.teleprompt.copro.task_specs import (
    BasicGenerateInstructionTaskSpec,
    GenerateInstructionGivenAttemptsTaskSpec,
)
from dspy.teleprompt.infer_rules_specs import rules_induction_task_spec
from dspy.teleprompt.simba_specs import SimbaOfferFeedbackTaskSpec

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
    DescribeProgramTaskSpec,
    DescribeModuleTaskSpec,
    ObservationSummarizerTaskSpec,
    DatasetDescriptorTaskSpec,
    DatasetDescriptorWithPriorObservationsTaskSpec,
]


@pytest.mark.parametrize("spec_cls", FRAMEWORK_SPECS)
def test_framework_task_specs_have_explicit_field_descs(spec_cls):
    spec = spec_cls()
    for field in (*spec.inputs, *spec.outputs):
        assert field.desc.strip(), f"{spec_cls.__name__}.{field.name} must have a non-empty desc"
    assert spec.name.startswith("framework.")


def test_generate_instruction_task_spec_has_explicit_field_descs():
    spec = generate_instruction_task_spec()
    for field in (*spec.inputs, *spec.outputs):
        assert field.desc.strip(), f"generate_instruction_task_spec.{field.name} must have a non-empty desc"
    assert spec.name == "framework.propose.generate_single_module_instruction"


def test_rules_induction_task_spec_has_explicit_field_descs():
    spec = rules_induction_task_spec(3)
    for field in (*spec.inputs, *spec.outputs):
        assert field.desc.strip(), f"rules_induction_task_spec.{field.name} must have a non-empty desc"
    assert spec.name == "framework.infer_rules.induction"


def test_two_step_extractor_task_spec_has_explicit_field_descs():
    original = make_task_spec("q -> a", instructions="Answer.")
    spec = build_extractor_task_spec(original, native_response_types=[])
    for field in (*spec.inputs, *spec.outputs):
        assert field.desc.strip(), f"build_extractor_task_spec.{field.name} must have a non-empty desc"
    assert spec.name == "framework.two_step.extractor"


def test_chain_of_thought_reasoning_field_has_explicit_desc():

    class QATaskSpec(TaskSpec):
        name: str = "QA"
        instructions: str = "Answer."
        inputs: tuple = (input_field("question", desc="Question."),)
        outputs: tuple = (output_field("answer", desc="Answer."),)

    cot = ChainOfThought(QATaskSpec())
    reasoning_field = cot.task_spec.output_fields["reasoning"]
    assert reasoning_field.desc.strip()
