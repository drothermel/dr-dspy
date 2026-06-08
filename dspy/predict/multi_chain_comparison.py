from dspy.predict.predict import Predict
from dspy.primitives.module import Module
from dspy.task_spec import TaskSpec, input_field, output_field


class MultiChainComparison(Module):
    def __init__(self, task_spec: TaskSpec, M=3, temperature=0.7, **config) -> None:
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
                input_field(field_name, str, desc="${reasoning attempt}", prefix=f"Student Attempt #{idx + 1}:")
            )
        extended_task_spec = extended_task_spec.prepend(
            output_field(
                "rationale",
                str,
                desc="${corrected reasoning}",
                prefix="Accurate Reasoning: Thank you everyone. Let's now holistically",
            )
        )
        self.predict = Predict(extended_task_spec, temperature=temperature, **config)

    async def aforward(self, completions, **kwargs):
        attempts = []
        for c in completions:
            rationale = c.get("rationale", c.get("reasoning")).strip().split("\n")[0].strip()
            answer = str(c[self.last_key]).strip().split("\n")[0].strip()
            attempts.append(f"«I'm trying to {rationale} I'm not sure but my prediction is {answer}»")
        assert len(attempts) == self.M, (
            f"The number of attempts ({len(attempts)}) doesn't match the expected number M ({self.M}). Please set the correct value for M when initializing MultiChainComparison."
        )
        kwargs = {**{f"reasoning_attempt_{idx + 1}": attempt for idx, attempt in enumerate(attempts)}, **kwargs}
        return await self.predict(**kwargs)
