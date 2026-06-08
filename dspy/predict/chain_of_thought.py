from typing import TYPE_CHECKING, Any, cast

from dspy.adapters.types.reasoning import Reasoning
from dspy.predict.predict import Predict
from dspy.primitives.module import Module
from dspy.signatures.field import OutputField
from dspy.signatures.signature import Signature, ensure_signature

if TYPE_CHECKING:
    from dspy.utils.callback import BaseCallback


class ChainOfThought(Module):
    def __init__(
        self,
        signature: str | type[Signature],
        **config: dict[str, Any],
    ) -> None:
        """
        A module that reasons step by step in order to predict the output of a task.

        Args:
            signature (Type[dspy.signatures.signature.Signature]): The signature of the module.
            **config: The configuration for the module.
        """
        super().__init__()
        signature = ensure_signature(signature)
        if signature is None:
            raise ValueError(f"Invalid signature: {signature!r}")
        extended_signature = signature.prepend(
            name="reasoning",
            field=OutputField(desc="${reasoning}"),
            type_=Reasoning,
        )
        callbacks = cast("list[BaseCallback] | None", config.pop("callbacks", None))
        self.predict = Predict(extended_signature, callbacks=callbacks, **config)

    async def aforward(self, **kwargs):
        return await self.predict(**kwargs)
