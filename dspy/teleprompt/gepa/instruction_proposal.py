import logging
from collections.abc import Mapping, Sequence
from typing import Any, cast

from typing_extensions import override

from dspy.adapters.types.base_type import Type
from dspy.predict.predict import Predict
from dspy.primitives.module import Module
from dspy.task_spec import FieldSpec, TaskSpec, input_field, output_field
from dspy.teleprompt.gepa.gepa_utils import AsyncProposalFn, ReflectiveExample

logger = logging.getLogger(__name__)


class GenerateEnhancedMultimodalInstructionTaskSpec(TaskSpec):
    name: str = "framework.gepa.multimodal_instruction"
    instructions: str = "I provided an assistant with instructions to perform a task involving visual content, but the assistant's performance needs improvement based on the examples and feedback below.\n\nYour task is to write a better instruction for the assistant that addresses the specific issues identified in the feedback, with particular attention to how visual and textual information should be analyzed and integrated.\n\n## Analysis Steps:\n1. **Read the inputs carefully** and identify both the visual and textual input formats, understanding how they work together\n2. **Read all the assistant responses and corresponding feedback** to understand what went wrong with visual analysis, text processing, or their integration\n3. **Identify visual analysis patterns** - what visual features, relationships, or details are important for this task\n4. **Identify domain-specific knowledge** about both visual and textual aspects, as this information may not be available to the assistant in the future\n5. **Look for successful visual-textual integration strategies** and include these patterns in the instruction\n6. **Address specific visual analysis issues** mentioned in the feedback\n\n## Instruction Requirements:\n- **Clear task definition** explaining how to process both visual and textual inputs\n- **Visual analysis guidance** specific to this task (what to look for, how to describe, what features matter)\n- **Integration strategies** for combining visual observations with textual information\n- **Domain-specific knowledge** about visual concepts, terminology, or relationships\n- **Error prevention guidance** for common visual analysis mistakes shown in the feedback\n- **Precise, actionable language** for both visual and textual processing\n\nFocus on creating an instruction that helps the assistant properly analyze visual content, integrate it with textual information, and avoid the specific visual analysis mistakes shown in the examples."
    inputs: tuple[FieldSpec, ...] = (
        input_field(
            "current_instruction",
            str,
            desc="The current instruction that was provided to the assistant to perform the multimodal task",
        ),
        input_field(
            "examples_with_feedback",
            str,
            desc="Task examples with visual content showing inputs, assistant outputs, and feedback. Pay special attention to feedback about visual analysis accuracy, visual-textual integration, and any domain-specific visual knowledge that the assistant missed.",
        ),
    )
    outputs: tuple[FieldSpec, ...] = (
        output_field(
            "improved_instruction",
            str,
            desc="A better instruction for the assistant that addresses visual analysis issues, provides clear guidance on how to process and integrate visual and textual information, includes necessary visual domain knowledge, and prevents the visual analysis mistakes shown in the examples.",
        ),
    )


class SingleComponentMultiModalProposer(Module):
    def __init__(self) -> None:
        super().__init__()
        self.propose_instruction = Predict(GenerateEnhancedMultimodalInstructionTaskSpec())

    async def aforward(self, current_instruction: str, reflective_dataset: list[ReflectiveExample]) -> str:
        formatted_examples, image_map = self._format_examples_with_pattern_analysis(reflective_dataset)
        predict_kwargs = {"current_instruction": current_instruction, "examples_with_feedback": formatted_examples}
        predict_kwargs["examples_with_feedback"] = self._create_multimodal_examples(
            formatted_text=formatted_examples, image_map=image_map
        )
        result = await self.propose_instruction(**predict_kwargs)
        return result.improved_instruction

    def _format_examples_with_pattern_analysis(
        self, reflective_dataset: list[ReflectiveExample]
    ) -> tuple[str, dict[int, list[Type]]]:
        formatted_examples, image_map = self._format_examples_for_instruction_generation(reflective_dataset)
        feedback_analysis = self._analyze_feedback_patterns(reflective_dataset)
        if feedback_analysis["summary"]:
            pattern_summary = self._create_pattern_summary(feedback_analysis)
            enhanced_examples = f"{pattern_summary}\n\n{formatted_examples}"
            return (enhanced_examples, image_map)
        return (formatted_examples, image_map)

    def _analyze_feedback_patterns(self, reflective_dataset: list[ReflectiveExample]) -> dict[str, Any]:
        analysis = {
            "error_patterns": [],
            "success_patterns": [],
            "domain_knowledge_gaps": [],
            "task_specific_guidance": [],
            "summary": "",
        }
        for example in reflective_dataset:
            feedback = example.get("Feedback", "").lower()
            if any(error_word in feedback for error_word in ["incorrect", "wrong", "error", "failed", "missing"]):
                analysis["error_patterns"].append(feedback)
            if any(
                success_word in feedback for success_word in ["correct", "good", "accurate", "well", "successfully"]
            ):
                analysis["success_patterns"].append(feedback)
            if any(
                knowledge_word in feedback
                for knowledge_word in ["should know", "domain", "specific", "context", "background"]
            ):
                analysis["domain_knowledge_gaps"].append(feedback)
        if any(analysis[key] for key in ["error_patterns", "success_patterns", "domain_knowledge_gaps"]):
            analysis["summary"] = (
                f"Patterns identified: {len(analysis['error_patterns'])} error(s), {len(analysis['success_patterns'])} success(es), {len(analysis['domain_knowledge_gaps'])} knowledge gap(s)"
            )
        return analysis

    def _create_pattern_summary(self, feedback_analysis: dict[str, Any]) -> str:
        summary_parts = ["## Feedback Pattern Analysis\n"]
        if feedback_analysis["error_patterns"]:
            summary_parts.append(f"**Common Issues Found ({len(feedback_analysis['error_patterns'])} examples):**")
            summary_parts.append("Focus on preventing these types of mistakes in the new instruction.\n")
        if feedback_analysis["success_patterns"]:
            summary_parts.append(
                f"**Successful Approaches Found ({len(feedback_analysis['success_patterns'])} examples):**"
            )
            summary_parts.append("Build on these successful strategies in the new instruction.\n")
        if feedback_analysis["domain_knowledge_gaps"]:
            summary_parts.append(
                f"**Domain Knowledge Needs Identified ({len(feedback_analysis['domain_knowledge_gaps'])} examples):**"
            )
            summary_parts.append("Include this specialized knowledge in the new instruction.\n")
        return "\n".join(summary_parts)

    def _format_examples_for_instruction_generation(
        self, reflective_dataset: list[ReflectiveExample]
    ) -> tuple[str, dict[int, list[Type]]]:

        def render_value_with_images(value, level=3, example_images=None):
            if example_images is None:
                example_images = []
            if isinstance(value, Type):
                image_idx = len(example_images) + 1
                example_images.append(value)
                return f"[IMAGE-{image_idx} - see visual content]\n\n"
            if isinstance(value, dict):
                s = ""
                for k, v in value.items():
                    s += f"{'#' * level} {k}\n"
                    s += render_value_with_images(v, level=min(level + 1, 6), example_images=example_images)
                if not value:
                    s += "\n"
                return s
            if isinstance(value, (list, tuple)):
                s = ""
                for i, item in enumerate(value):
                    s += f"{'#' * level} Item {i + 1}\n"
                    s += render_value_with_images(item, level=min(level + 1, 6), example_images=example_images)
                if not value:
                    s += "\n"
                return s
            return f"{str(value).strip()}\n\n"

        def convert_sample_to_markdown_with_images(sample, example_num):
            example_images = []
            s = f"# Example {example_num}\n"
            for key, val in sample.items():
                s += f"## {key}\n"
                s += render_value_with_images(val, level=3, example_images=example_images)
            return (s, example_images)

        formatted_parts = []
        image_map = {}
        for i, example_data in enumerate(reflective_dataset):
            formatted_example, example_images = convert_sample_to_markdown_with_images(
                sample=example_data, example_num=i + 1
            )
            formatted_parts.append(formatted_example)
            if example_images:
                image_map[i] = example_images
        formatted_text = "\n\n".join(formatted_parts)
        if image_map:
            total_images = sum(len(imgs) for imgs in image_map.values())
            formatted_text = (
                f"The examples below include visual content ({total_images} images total). Please analyze both the text and visual elements when suggesting improvements.\n\n"
                + formatted_text
            )
        return (formatted_text, image_map)

    def _create_multimodal_examples(self, formatted_text: str, image_map: dict[int, list[Type]]) -> Any:
        if not image_map:
            return formatted_text
        all_images = []
        for example_images in image_map.values():
            all_images.extend(example_images)
        multimodal_content: list[Any] = [formatted_text]
        multimodal_content.extend(cast("list[Any]", all_images))
        return multimodal_content


class MultiModalInstructionProposer(AsyncProposalFn):
    def __init__(self) -> None:
        self.single_proposer = SingleComponentMultiModalProposer()

    @override
    async def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        updated_components = {}
        for component_name in components_to_update:
            if component_name in candidate and component_name in reflective_dataset:
                current_instruction = candidate[component_name]
                component_reflective_data = cast("list[ReflectiveExample]", reflective_dataset[component_name])
                new_instruction = await self.single_proposer(
                    current_instruction=current_instruction, reflective_dataset=component_reflective_data
                )
                updated_components[component_name] = new_instruction
        return updated_components
