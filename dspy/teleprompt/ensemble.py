import random

from pydantic import BaseModel

from dspy.core.types.call_options import ModuleCallOptions
from dspy.primitives import Module
from dspy.runtime.run_context import RunContext, resolve_run
from dspy.teleprompt.compilation import CompileResult
from dspy.teleprompt.compile_params import EnsembleCompileParams
from dspy.teleprompt.registry import register_teleprompter


@register_teleprompter(params=EnsembleCompileParams)
class Ensemble:
    def __init__(self, *, reduce_fn=None, size=None, deterministic=False) -> None:
        assert deterministic is False, "Deterministic ensemble is not supported; Example is intentionally unhashable."
        self.reduce_fn = reduce_fn
        self.size = size
        self.deterministic = deterministic

    async def compile(self, student: Module, *, params: BaseModel, run: RunContext) -> CompileResult:
        params = EnsembleCompileParams.model_validate(params)
        programs = params.programs
        size = self.size
        reduce_fn = self.reduce_fn

        class EnsembledProgram(Module):
            def __init__(self) -> None:
                super().__init__()
                self.programs = programs

            async def _aforward_impl(
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

        return CompileResult(program=EnsembledProgram())
