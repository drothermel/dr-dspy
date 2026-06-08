import json
import tempfile
from pathlib import Path

import pytest

from dspy.dsp.utils.settings import settings
from dspy.predict.chain_of_thought import ChainOfThought
from dspy.predict.predict import Predict
from dspy.primitives.example import Example
from dspy.primitives.module import Module, set_attribute_by_name
from dspy.signatures.field import InputField, OutputField
from dspy.signatures.signature import Signature
from dspy.utils.dummies import DummyLM


class HopModule(Module):
    def __init__(self):
        super().__init__()
        self.predict1 = Predict("question -> query")
        self.predict2 = Predict("query -> answer")

    def forward(self, question):
        query = self.predict1(question=question).query
        return self.predict2(query=query)


def test_module_initialization():
    module = Module()
    assert module._compiled is False, "Module _compiled attribute should be False upon initialization"


def test_named_predictors():
    module = HopModule()
    named_preds = module.named_predictors()
    assert len(named_preds) == 2, "Should identify correct number of Predict instances"
    names, _preds = zip(*named_preds, strict=False)
    assert "self.predict1" in names and "self.predict2" in names, (
        "Named predictors should include 'self.predict1' and 'self.predict2'"
    )


def test_predictors():
    module = HopModule()
    preds = module.predictors()
    assert len(preds) == 2, "Should return correct number of Predict instances"
    assert all(isinstance(p, Predict) for p in preds), "All returned items should be instances of PredictMock"


def test_forward():
    program = HopModule()
    settings.configure(
        lm=DummyLM(
            {
                "What is 1+1?": {"query": "let me check"},
                "let me check": {"answer": "2"},
            }
        )
    )
    result = program(question="What is 1+1?").answer
    assert result == "2"


def test_nested_named_predictors():
    class Hop2Module(Module):
        def __init__(self):
            super().__init__()
            self.hop = HopModule()

    module = Hop2Module()
    named_preds = module.named_predictors()
    assert len(named_preds) == 2
    names, _preds = zip(*named_preds, strict=False)
    assert "self.hop.predict1" in names
    assert "self.hop.predict2" in names


def test_empty_module():
    module = Module()
    assert list(module.named_sub_modules()) == [("self", module)]


def test_single_level():
    module = Module()
    module.sub = Module()  # ty:ignore[unresolved-attribute]
    expected = [("self", module), ("self.sub", module.sub)]
    assert list(module.named_sub_modules()) == expected


def test_multiple_levels():
    module = Module()
    module.sub = Module()  # ty:ignore[unresolved-attribute]
    module.sub.subsub = Module()  # ty:ignore[unresolved-attribute]
    expected = [("self", module), ("self.sub", module.sub), ("self.sub.subsub", module.sub.subsub)]
    assert list(module.named_sub_modules()) == expected


def test_multiple_sub_modules():
    module = Module()
    module.sub1 = Module()  # ty:ignore[unresolved-attribute]
    module.sub2 = Module()  # ty:ignore[unresolved-attribute]
    expected = [("self", module), ("self.sub1", module.sub1), ("self.sub2", module.sub2)]
    assert sorted(module.named_sub_modules()) == sorted(expected)


def test_non_base_module_attributes():
    module = Module()
    module.sub = Module()  # ty:ignore[unresolved-attribute]
    module.not_a_sub = "Not a self"  # ty:ignore[unresolved-attribute]
    expected = [("self", module), ("self.sub", module.sub)]
    assert list(module.named_sub_modules()) == expected


def test_complex_module_traversal():
    root = Module()
    root.sub_module = Module()  # ty:ignore[unresolved-attribute]
    root.sub_module.nested_list = [Module(), {"key": Module()}]  # ty:ignore[unresolved-attribute]
    root.sub_module.nested_tuple = (Module(), [Module(), Module()])  # ty:ignore[unresolved-attribute]
    expected_names = {
        "self",
        "self.sub_module",
        "self.sub_module.nested_list[0]",
        "self.sub_module.nested_list[1][key]",
        "self.sub_module.nested_tuple[0]",
        "self.sub_module.nested_tuple[1][0]",
        "self.sub_module.nested_tuple[1][1]",
    }
    found_names = {name for name, _ in root.named_sub_modules()}

    assert found_names == expected_names, (
        f"Missing or extra modules found. Missing: {expected_names - found_names}, Extra: {found_names - expected_names}"
    )


def test_complex_module_traversal_with_same_module():
    root = Module()
    root.sub_module = Module()  # ty:ignore[unresolved-attribute]
    root.sub_module.nested_list = [Module(), {"key": Module()}]  # ty:ignore[unresolved-attribute]
    same_module = Module()
    root.sub_module.nested_tuple = (Module(), [same_module, same_module])  # ty:ignore[unresolved-attribute]
    expected_names = {
        "self",
        "self.sub_module",
        "self.sub_module.nested_list[0]",
        "self.sub_module.nested_list[1][key]",  # NOTE: named_sub_modules allows recursive structures
        "self.sub_module.nested_tuple[0]",
        "self.sub_module.nested_tuple[1][0]",
    }
    found_names = {name for name, _ in root.named_sub_modules()}

    assert found_names == expected_names, (
        f"Missing or extra modules found. Missing: {expected_names - found_names}, Extra: {found_names - expected_names}"
    )


def test_named_parameters_traverses_nested_containers():
    root = Module()
    root.sub_module = Module()  # ty:ignore[unresolved-attribute]
    root.sub_module.nested_predict = Predict("question -> answer")  # ty:ignore[unresolved-attribute]
    root.sub_module.nested_list = [Predict("question -> answer")]  # ty:ignore[unresolved-attribute]
    root.sub_module.nested_dict = {"key": Predict("question -> answer")}  # ty:ignore[unresolved-attribute]

    found_names = {name for name, _ in root.named_parameters()}

    assert "self.sub_module.nested_predict" in found_names
    assert "self.sub_module.nested_list[0]" in found_names
    assert "self.sub_module.nested_dict[key]" in found_names


def test_complex_module_set_attribute_by_name():
    root = Module()
    root.sub_module = Module()  # ty:ignore[unresolved-attribute]
    root.sub_module.nested_list = [Module(), {"key": Module()}]  # ty:ignore[unresolved-attribute]
    same_module = Module()
    root.sub_module.nested_tuple = (Module(), [same_module, same_module])  # ty:ignore[unresolved-attribute]

    set_attribute_by_name(root, "test_attrib", True)
    assert root.test_attrib is True
    set_attribute_by_name(root, "sub_module.test_attrib", True)
    assert root.sub_module.test_attrib is True
    set_attribute_by_name(root, "sub_module.nested_list[0].test_attrib", True)
    assert root.sub_module.nested_list[0].test_attrib is True  # ty:ignore[unresolved-attribute]
    set_attribute_by_name(root, "sub_module.nested_list[1]['key'].test_attrib", True)
    assert root.sub_module.nested_list[1]["key"].test_attrib is True  # ty:ignore[not-subscriptable]
    set_attribute_by_name(root, "sub_module.nested_tuple[0].test_attrib", True)
    assert root.sub_module.nested_tuple[0].test_attrib is True
    set_attribute_by_name(root, "sub_module.nested_tuple[1][0].test_attrib", True)
    assert root.sub_module.nested_tuple[1][0].test_attrib is True
    assert root.sub_module.nested_tuple[1][1].test_attrib is True


class DuplicateModule(Module):
    def __init__(self):
        super().__init__()
        self.p0 = Predict("question -> answer")
        self.p1 = self.p0


def test_named_parameters_duplicate_references():
    module = DuplicateModule()
    # Only testing for whether exceptions are thrown or not
    # As Module.named_parameters() is recursive, this is mainly for catching infinite recursion
    module.named_parameters()


def test_load_state_is_transactional():
    """
    Regression test for https://github.com/stanfordnlp/dspy/issues/9589

    load_state must be all-or-nothing. If it fails mid-load (missing key
    or malformed value), the module must be completely unchanged.
    """

    class Sig(Signature):
        question: str = InputField()
        answer: str = OutputField()

    class Prog(Module):
        def __init__(self):
            super().__init__()
            self.a = ChainOfThought(Sig)
            self.b = ChainOfThought(Sig)

    source = Prog()
    sentinel = Example(question="q1", answer="a1").with_inputs("question")
    source.a.predict.demos = [sentinel]
    source.b.predict.demos = [sentinel]

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "state.json"
        source.save(str(path), save_program=False)

        raw = json.loads(path.read_text())
        corrupted = {k: v for k, v in raw.items() if "b." not in k}
        path.write_text(json.dumps(corrupted))

        template = Prog()
        assert template.a.predict.demos == []

        with pytest.raises(KeyError):
            template.load(str(path))

        assert template.a.predict.demos == [], "load_state partially mutated module before failing"
