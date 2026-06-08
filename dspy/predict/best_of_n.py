from collections.abc import Callable

from dspy.dsp.utils.settings import settings
from dspy.predict.predict import Module, Prediction


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

    async def aforward(self, **kwargs):
        lm = self.module.get_lm() or settings.lm
        best_pred, best_trace, best_reward = (None, None, -float("inf"))
        for idx in range(self.N):
            lm_ = lm.copy(temperature=1.0)
            mod = self.module.deepcopy()
            mod.set_lm(lm_)
            try:
                with settings.context(trace=[]):
                    pred = await mod(**kwargs)
                    trace = settings.trace.copy()
                    reward = self.reward_fn(kwargs, pred)
                if reward > best_reward:
                    best_reward, best_pred, best_trace = (reward, pred, trace)
                if reward >= self.threshold:
                    break
            except Exception:
                if idx > self.fail_count:
                    raise
                self.fail_count -= 1
        if best_trace:
            settings.trace.extend(best_trace)
        return best_pred
