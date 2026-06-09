import pytest

from dspy.predict.predict import Predict
from dspy.primitives import Example, Prediction
from dspy.teleprompt.errors import UnknownPredictorInTraceError
from dspy.teleprompt.trace_helpers import trace_to_demos
from tests.task_spec.helpers import ts


def test_trace_to_demos_raises_on_unknown_predictor():
    predictor = Predict(ts("input -> output"))
    example = Example.from_record({"input": "x", "output": "y"}, input_keys=("input",))
    trace = [(predictor, {"input": example.input}, Prediction(output=example.output))]
    with pytest.raises(UnknownPredictorInTraceError, match="No predictor mapping"):
        trace_to_demos(trace, predictor2name={})
