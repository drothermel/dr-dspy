from typing import TYPE_CHECKING, Any, cast

from dspy.adapters.types.reasoning import Reasoning
from dspy.predict.predict import Predict
from dspy.primitives.module import Module
from dspy.task_spec import FieldSpec, TaskSpec

if TYPE_CHECKING:
    from dspy.utils.callback import BaseCallback


class ChainOfThought(Module):
    def __init__(
        self,
        task_spec: TaskSpec,
        **config: dict[str, Any],
    ) -> None:
        """
        A module that reasons step by step in order to predict the output of a task.

        Args:
            task_spec: The task spec for the module.
            **config: The configuration for the module.
        """
        super().__init__()
        if not isinstance(task_spec, TaskSpec):
            raise TypeError(f"ChainOfThought requires a TaskSpec instance, got {type(task_spec).__name__}.")
        extended_task_spec = task_spec.prepend(
            FieldSpec.output("reasoning", Reasoning, desc="${reasoning}"),
        )
        callbacks = cast("list[BaseCallback] | None", config.pop("callbacks", None))
        self.task_spec = task_spec
        self.predict = Predict(extended_task_spec, callbacks=callbacks, **config)

    async def aforward(self, **kwargs):
        return await self.predict(**kwargs)
