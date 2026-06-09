from collections.abc import Callable

from dspy.predict.sampling import sample_with_reward
from dspy.primitives import Module, Prediction
from dspy.runtime.call_options import ModuleCallOptions
from dspy.runtime.run_context import RunContext, resolve_run


class BestOfN(Module):
    def __init__(
        self,
        module: Module,
        num_samples: int,
        reward_fn: Callable[[dict, Prediction], float],
        threshold: float,
        fail_count: int | None = None,
    ) -> None:
        super().__init__()
        self.module = module
        self.reward_fn = lambda *args: reward_fn(*args)
        self.threshold = threshold
        self.num_samples = num_samples
        self.fail_count = fail_count or num_samples

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
            num_samples=self.num_samples,
            fail_count=self.fail_count,
            reward_fn=self.reward_fn,
            threshold=self.threshold,
            run=run,
            options=options,
            inputs=inputs,
            should_stop=lambda _attempt, reward, _state: reward >= self.threshold,
        )
