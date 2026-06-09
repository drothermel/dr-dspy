from __future__ import annotations

from typing import Any

from dspy.core.types import LMConfig, merge_lm_config
from dspy.predict.predict import Predict
from dspy.primitives import Module, Prediction
from dspy.runtime.call_options import ModuleCallOptions  # noqa: TC001 — runtime signature typing
from dspy.runtime.run_context import RunContext  # noqa: TC001 — runtime signature typing
from dspy.task_spec import TaskSpec, input_field, output_field

StudentCompletion = Prediction | dict[str, Any]


class MultiChainComparison(Module):
    """Compare multiple student chain completions and synthesize corrected reasoning.

    Callers must pass ``student_completions`` as a keyword argument: a list of
    ``Prediction`` or mapping records containing each chain's rationale/reasoning
    and final output field. The list length must equal ``num_chains``.
    """

    def __init__(
        self,
        task_spec: TaskSpec,
        num_chains: int = 3,
        *,
        config: LMConfig | None = None,
        temperature: float = 0.7,
    ) -> None:
        super().__init__()
        if not isinstance(task_spec, TaskSpec):
            raise TypeError(f"MultiChainComparison requires a TaskSpec instance, got {type(task_spec).__name__}.")
        self.num_chains = num_chains
        self.task_spec = task_spec
        *_, self.last_key = task_spec.output_fields.keys()
        extended_task_spec = task_spec
        for idx in range(num_chains):
            field_name = f"reasoning_attempt_{idx + 1}"
            extended_task_spec = extended_task_spec.append(
                input_field(
                    field_name,
                    str,
                    desc=f"Reasoning attempt {idx + 1} from a student chain.",
                    prefix=f"Student Attempt #{idx + 1}:",
                )
            )
        extended_task_spec = extended_task_spec.prepend(
            output_field(
                "rationale",
                str,
                desc="Corrected reasoning synthesized from student attempts.",
                prefix="Accurate Reasoning: Thank you everyone. Let's now holistically",
            )
        )
        merged_config = merge_lm_config(LMConfig(temperature=temperature), config) or LMConfig(temperature=temperature)
        self.predict = Predict(extended_task_spec, config=merged_config)

    async def _aforward_impl(
        self,
        *,
        run: RunContext,
        options: ModuleCallOptions | None = None,
        student_completions: list[StudentCompletion],
        **inputs: Any,
    ):
        if not student_completions:
            raise TypeError("MultiChainComparison requires student_completions: list[Prediction | dict[str, Any]].")
        attempts = []
        for completion in student_completions:
            rationale_source = (
                completion.get("rationale", completion.get("reasoning"))
                if hasattr(completion, "get")
                else getattr(completion, "rationale", getattr(completion, "reasoning", ""))
            )
            rationale = str(rationale_source).strip().split("\n")[0].strip()
            answer_value = (
                completion[self.last_key] if hasattr(completion, "__getitem__") else getattr(completion, self.last_key)
            )
            answer = str(answer_value).strip().split("\n")[0].strip()
            attempts.append(f"«I'm trying to {rationale} I'm not sure but my prediction is {answer}»")
        if len(attempts) != self.num_chains:
            raise ValueError(
                f"The number of attempts ({len(attempts)}) doesn't match num_chains ({self.num_chains}). "
                "Set num_chains when initializing MultiChainComparison."
            )
        merged_inputs = {
            **{f"reasoning_attempt_{idx + 1}": attempt for idx, attempt in enumerate(attempts)},
            **inputs,
        }
        return await self.predict(run=run, options=options, **merged_inputs)
