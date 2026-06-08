from dspy.predict.predict import Predict
from dspy.primitives.module import Module
from dspy.signatures.field import InputField, OutputField
from dspy.signatures.signature import Signature, ensure_signature


class MultiChainComparison(Module):
    def __init__(self, signature: str | type[Signature], M=3, temperature=0.7, **config) -> None:  # noqa: N803
        super().__init__()

        self.M = M
        signature = ensure_signature(signature)
        if signature is None:
            raise ValueError("Invalid signature provided to MultiChainComparison.")

        *_, self.last_key = signature.output_fields.keys()

        for idx in range(M):
            field_name = f"reasoning_attempt_{idx + 1}"
            signature = signature.append(
                field_name,
                InputField(desc="${reasoning attempt}"),
            ).with_updated_fields(field_name, prefix=f"Student Attempt #{idx + 1}:")

        signature = signature.prepend(
            "rationale",
            OutputField(desc="${corrected reasoning}"),
        ).with_updated_fields(
            "rationale",
            prefix="Accurate Reasoning: Thank you everyone. Let's now holistically",
        )

        self.predict = Predict(signature, temperature=temperature, **config)

    async def aforward(self, completions, **kwargs):
        attempts = []

        for c in completions:
            rationale = c.get("rationale", c.get("reasoning")).strip().split("\n")[0].strip()
            answer = str(c[self.last_key]).strip().split("\n")[0].strip()
            attempts.append(
                f"«I'm trying to {rationale} I'm not sure but my prediction is {answer}»",
            )

        assert len(attempts) == self.M, (
            f"The number of attempts ({len(attempts)}) doesn't match the expected number M ({self.M}). Please set the correct value for M when initializing MultiChainComparison."
        )

        kwargs = {
            **{f"reasoning_attempt_{idx + 1}": attempt for idx, attempt in enumerate(attempts)},
            **kwargs,
        }
        return await self.predict(**kwargs)
