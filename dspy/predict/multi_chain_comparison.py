from dspy.core.types import LMConfig, merge_lm_config
from dspy.core.types.call_options import ModuleCallOptions
from dspy.predict.predict import Predict
from dspy.primitives.module import Module
from dspy.runtime.run_context import RunContext
from dspy.task_spec import TaskSpec, input_field, output_field


class MultiChainComparison(Module):
    def __init__(
        self,
        task_spec: TaskSpec,
        M: int = 3,
        *,
        config: LMConfig | None = None,
        temperature: float = 0.7,
    ) -> None:
        super().__init__()
        if not isinstance(task_spec, TaskSpec):
            raise TypeError(f"MultiChainComparison requires a TaskSpec instance, got {type(task_spec).__name__}.")
        self.M = M
        self.task_spec = task_spec
        *_, self.last_key = task_spec.output_fields.keys()
        extended_task_spec = task_spec
        for idx in range(M):
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
        completions: list,
        **inputs,
    ):
        attempts = []
        for c in completions:
            rationale = c.get("rationale", c.get("reasoning")).strip().split("\n")[0].strip()
            answer = str(c[self.last_key]).strip().split("\n")[0].strip()
            attempts.append(f"«I'm trying to {rationale} I'm not sure but my prediction is {answer}»")
        assert len(attempts) == self.M, (
            f"The number of attempts ({len(attempts)}) doesn't match the expected number M ({self.M}). Please set the correct value for M when initializing MultiChainComparison."
        )
        merged_inputs = {
            **{f"reasoning_attempt_{idx + 1}": attempt for idx, attempt in enumerate(attempts)},
            **inputs,
        }
        return await self.predict(run=run, options=options, **merged_inputs)
