from dspy.clients.lm import LM
from dspy.predict.predict import Predict
from dspy.teleprompt.task_spec_context import resolve_optimizer_lm


def test_resolve_optimizer_lm_uses_run_default(make_run):
    _ = Predict
    run = make_run(lm=LM("openai/gpt-4o-mini", temperature=0.0, max_tokens=100))
    assert resolve_optimizer_lm(None, run=run) is run.lm


def test_resolve_optimizer_lm_preserves_explicit_lm(make_run):
    _ = Predict
    run = make_run(lm=LM("openai/gpt-4o-mini", temperature=0.0, max_tokens=100))
    explicit = LM("openai/gpt-4.1-mini", temperature=0.5, max_tokens=200)
    assert resolve_optimizer_lm(explicit, run=run) is explicit
