from dspy.clients.base_lm import BaseLM
from dspy.runtime.run_context import RunContext


def resolve_optimizer_lm(lm: BaseLM | None, *, run: RunContext) -> BaseLM:
    if lm is not None:
        return lm
    return run.lm


def get_task_spec(predictor):
    assert hasattr(predictor, "task_spec")
    return predictor.task_spec


def set_task_spec(*, predictor, task_spec) -> None:
    assert hasattr(predictor, "task_spec")
    predictor.task_spec = task_spec
