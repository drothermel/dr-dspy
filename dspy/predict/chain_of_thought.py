from dspy.adapters.types.reasoning import Reasoning
from dspy.core.types.call_options import ModuleCallOptions
from dspy.core.types.config import LMConfig
from dspy.predict.predict import Predict
from dspy.primitives import Module, Prediction
from dspy.runtime.callback import Callback
from dspy.runtime.run_context import RunContext
from dspy.task_spec import TaskSpec, output_field


class ChainOfThought(Module):
    def __init__(
        self,
        task_spec: TaskSpec,
        *,
        config: LMConfig | None = None,
        callbacks: list[Callback] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(task_spec, TaskSpec):
            raise TypeError(f"ChainOfThought requires a TaskSpec instance, got {type(task_spec).__name__}.")
        extended_task_spec = task_spec.prepend(
            output_field("reasoning", Reasoning, desc="Step-by-step reasoning before producing the final outputs.")
        )
        self.task_spec = extended_task_spec
        self.predict = Predict(extended_task_spec, config=config, callbacks=callbacks)

    async def _aforward_impl(
        self,
        *,
        run: RunContext,
        options: ModuleCallOptions | None = None,
        **inputs,
    ) -> Prediction:
        return await self.predict(run=run, options=options, **inputs)
