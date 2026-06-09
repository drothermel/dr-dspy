from collections.abc import Callable

from dspy.core.types.call_options import ModuleCallOptions
from dspy.predict.predict import Module, Prediction
from dspy.runtime.run_context import RunContext, resolve_run
from dspy.teleprompt.trace_helpers import run_program_with_trace


class BestOfN(Module):
    def __init__(
        self,
        module: Module,
        N: int,
        reward_fn: Callable[[dict, Prediction], float],
        threshold: float,
        fail_count: int | None = None,
    ) -> None:
        self.module = module
        self.reward_fn = lambda *args: reward_fn(*args)
        self.threshold = threshold
        self.N = N
        self.fail_count = fail_count or N

    async def _aforward_impl(
        self,
        *,
        run: RunContext,
        options: ModuleCallOptions | None = None,
        **inputs,
    ):
        run = resolve_run(run=run, bound_run=self.run)
        lm = self.module.optional_lm() or run.lm
        best_pred, best_trace, best_reward = (None, None, -float("inf"))
        for idx in range(self.N):
            lm_ = lm.copy(temperature=1.0)
            mod = self.module.deepcopy()
            mod.set_lm(lm_)
            try:
                pred, trace = await run_program_with_trace(mod, inputs, run, options=options)
                reward = self.reward_fn(inputs, pred)
                if reward > best_reward:
                    best_reward, best_pred, best_trace = (reward, pred, trace)
                if reward >= self.threshold:
                    break
            except Exception:
                if idx > self.fail_count:
                    raise
                self.fail_count -= 1
        if best_trace:
            run.optimization_trace.extend(best_trace)
        return best_pred
