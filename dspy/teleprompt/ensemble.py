import random

from typing_extensions import override

from dspy.core.types.call_options import ModuleCallOptions
from dspy.primitives.module import Module
from dspy.runtime.run_context import RunContext, resolve_run
from dspy.teleprompt.teleprompt import Teleprompter


class Ensemble(Teleprompter):
    def __init__(self, *, reduce_fn=None, size=None, deterministic=False) -> None:
        assert deterministic is False, "TODO: Implement example hashing for deterministic ensemble."
        self.reduce_fn = reduce_fn
        self.size = size
        self.deterministic = deterministic

    @override
    async def compile(self, programs, *, run: RunContext):
        size = self.size
        reduce_fn = self.reduce_fn

        class EnsembledProgram(Module):
            def __init__(self) -> None:
                super().__init__()
                self.programs = programs

            async def aforward(
                self,
                *,
                run: RunContext,
                options: ModuleCallOptions | None = None,
                **inputs,
            ):
                run = resolve_run(run=run, bound_run=self.run)
                programs = random.sample(self.programs, size) if size else self.programs
                outputs = [await prog(run=run, options=options, **inputs) for prog in programs]
                if reduce_fn:
                    return reduce_fn(outputs)
                return outputs

        return EnsembledProgram()
