from dspy.predict.predict import Predict
from dspy.primitives import Example, Module
from dspy.teleprompt.bootstrap import BootstrapFewShot
from dspy.teleprompt.bootstrap_session import BootstrapCompileSession
from tests.task_spec.helpers import ts


class _TwoPredictorModule(Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = Predict(ts("question -> answer"))
        self.second = Predict(ts("question -> answer"))


def test_bootstrap_train_preserves_shared_labeled_demo_pool():
    validation = [Example.from_record({"question": f"q{i}", "answer": f"a{i}"}) for i in range(5)]
    student = _TwoPredictorModule()
    session = BootstrapCompileSession(
        student=student,
        teacher=student,
        trainset=validation,
        validation=validation,
        name2traces={"self.first": [], "self.second": []},
        name2predictor={"self.first": student.first, "self.second": student.second},
        predictor2name={id(student.first): "first", id(student.second): "second"},
    )
    BootstrapFewShot(max_labeled_demos=2, max_bootstrapped_demos=0)._train(session)
    assert len(student.first.demos) == 2
    assert len(student.second.demos) == 2
