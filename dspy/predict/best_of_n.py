from collections.abc import Callable

from dspy.core.types.call_options import ModuleCallOptions
from dspy.predict.predict import Module, Prediction
from dspy.predict.sampling import sample_with_reward
from dspy.runtime.run_context import RunContext, resolve_run


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
        return await sample_with_reward(
            module=self.module,
            N=self.N,
            fail_count=self.fail_count,
            reward_fn=self.reward_fn,
            threshold=self.threshold,
            run=run,
            options=options,
            inputs=inputs,
            should_stop=lambda _attempt, reward, _state: reward >= self.threshold,
        )
