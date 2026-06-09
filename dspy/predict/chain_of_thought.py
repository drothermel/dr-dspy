from typing import TYPE_CHECKING

from dspy.adapters.types.reasoning import Reasoning
from dspy.core.types.call_options import ModuleCallOptions
from dspy.core.types.config import LMConfig
from dspy.predict.predict import Predict
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.runtime.run_context import RunContext
from dspy.task_spec import TaskSpec, output_field

if TYPE_CHECKING:
    from dspy.utils.callback import BaseCallback


class ChainOfThought(Module):
    def __init__(
        self,
        task_spec: TaskSpec,
        *,
        config: LMConfig | None = None,
        callbacks: list["BaseCallback"] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(task_spec, TaskSpec):
            raise TypeError(f"ChainOfThought requires a TaskSpec instance, got {type(task_spec).__name__}.")
        extended_task_spec = task_spec.prepend(
            output_field("reasoning", Reasoning, desc="Step-by-step reasoning before producing the final outputs.")
        )
        self.task_spec = task_spec
        self.predict = Predict(extended_task_spec, config=config, callbacks=callbacks)

    async def aforward(
        self,
        *,
        run: RunContext,
        options: ModuleCallOptions | None = None,
        **inputs,
    ) -> Prediction:
        return await self.predict(run=run, options=options, **inputs)
