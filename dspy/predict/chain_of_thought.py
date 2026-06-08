from typing import TYPE_CHECKING, Any, cast

from pydantic.fields import FieldInfo

from dspy.predict.predict import Predict
from dspy.primitives.module import Module
from dspy.signatures.field import OutputField
from dspy.signatures.signature import Signature, ensure_signature

if TYPE_CHECKING:
    from dspy.utils.callback import BaseCallback

# NOTE: This restores the legacy rationale_field behavior after PR #8822.


class ChainOfThought(Module):
    def __init__(
        self,
        signature: str | type[Signature],
        rationale_field: FieldInfo | None = None,
        rationale_field_type: type = str,
        **config: dict[str, Any],
    ) -> None:
        """
        A module that reasons step by step in order to predict the output of a task.

        Args:
            signature (Type[dspy.signatures.signature.Signature]): The signature of the module.
            rationale_field (Optional[Union[dspy.signatures.field.OutputField, pydantic.fields.FieldInfo]]): The field that will contain the reasoning.
            rationale_field_type (Type): The type of the rationale field.
            **config: The configuration for the module.
        """
        super().__init__()
        signature = ensure_signature(signature)
        if signature is None:
            raise ValueError(f"Invalid signature: {signature!r}")
        desc = "${reasoning}"
        rationale_field_type = cast("type", rationale_field.annotation) if rationale_field else rationale_field_type
        rationale_field = rationale_field if rationale_field else OutputField(desc=desc)
        extended_signature = signature.prepend(name="reasoning", field=rationale_field, type_=rationale_field_type)
        callbacks = cast("list[BaseCallback] | None", config.pop("callbacks", None))
        self.predict = Predict(extended_signature, callbacks=callbacks, **config)

    def forward(self, **kwargs):
        return self.predict(**kwargs)

    async def aforward(self, **kwargs):
        return await self.predict.acall(**kwargs)
