from dspy.adapters.call.wrappers import HintInjectingAdapter
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.predict.predict import Predict
from dspy.primitives import Module
from dspy.runtime.config import CallSite
from tests.task_spec.helpers import ts


class _TwoPredictorModule(Module):
    def __init__(self) -> None:
        super().__init__()
        shared_spec = ts("question -> answer")
        self.first = Predict(shared_spec)
        self.second = Predict(shared_spec)


def test_refine_predictor_names_use_identity_not_shared_task_spec():
    module = _TwoPredictorModule()
    predictor_id_to_name = {id(predictor): name for name, predictor in module.named_predictors()}
    for name, predictor in module.named_predictors():
        object.__setattr__(predictor, "_dspy_predictor_name", name)
    assert predictor_id_to_name[id(module.first)] == "self.first"
    assert predictor_id_to_name[id(module.second)] == "self.second"
    adapter = HintInjectingAdapter(inner=ChatAdapter(), hint_map={"self.first": "hint-a", "self.second": "hint-b"})
    call_site_first = CallSite(module="Predict", predictor_name="self.first")
    call_site_second = CallSite(module="Predict", predictor_name="self.second")
    assert call_site_first.predictor_name is not None
    assert call_site_second.predictor_name is not None
    assert adapter._hint_map[call_site_first.predictor_name] == "hint-a"
    assert adapter._hint_map[call_site_second.predictor_name] == "hint-b"
