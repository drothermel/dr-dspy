import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, cast

import pytest

from dspy.predict.chain_of_thought import ChainOfThought
from dspy.predict.predict import Predict
from dspy.primitives.example import Example
from dspy.primitives.module import Module, set_attribute_by_name
from dspy.task_spec import default_task_instructions
from dspy.utils.dummies import DummyLM
from tests.task_spec.helpers import ts


def _module_attrs(module: Module) -> Any:
    return cast("Any", module)


class HopModule(Module):
    def __init__(self):
        super().__init__()
        self.predict1 = Predict(
            ts("question -> query", instructions=default_task_instructions(inputs=("question",), outputs=("query",)))
        )
        self.predict2 = Predict(
            ts("query -> answer", instructions=default_task_instructions(inputs=("query",), outputs=("answer",)))
        )

    async def aforward(self, question, **kwargs):
        run = kwargs.get("run")
        query = (await self.predict1(question=question, run=run)).query
        return await self.predict2(query=query, run=run)


def test_module_initialization(make_run):
    module = Module()
    assert module._compiled is False, "Module _compiled attribute should be False upon initialization"


def test_named_predictors(make_run):
    module = HopModule()
    named_preds = module.named_predictors()
    assert len(named_preds) == 2, "Should identify correct number of Predict instances"
    names, _preds = zip(*named_preds, strict=False)
    assert "self.predict1" in names and "self.predict2" in names, (
        "Named predictors should include 'self.predict1' and 'self.predict2'"
    )


def test_predictors(make_run):
    module = HopModule()
    preds = module.predictors()
    assert len(preds) == 2, "Should return correct number of Predict instances"
    assert all(isinstance(p, Predict) for p in preds), "All returned items should be instances of PredictMock"


def test_forward(make_run):
    program = HopModule()
    run = make_run(lm=DummyLM({"What is 1+1?": {"query": "let me check"}, "let me check": {"answer": "2"}}))
    result = asyncio.run(program(question="What is 1+1?", run=run)).answer
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
    _module_attrs(module).sub = Module()
    expected = [("self", module), ("self.sub", _module_attrs(module).sub)]
    assert list(module.named_sub_modules()) == expected


def test_multiple_levels():
    module = Module()
    _module_attrs(module).sub = Module()
    _module_attrs(module).sub.subsub = Module()
    expected = [
        ("self", module),
        ("self.sub", _module_attrs(module).sub),
        ("self.sub.subsub", _module_attrs(module).sub.subsub),
    ]
    assert list(module.named_sub_modules()) == expected


def test_multiple_sub_modules():
    module = Module()
    _module_attrs(module).sub1 = Module()
    _module_attrs(module).sub2 = Module()
    expected = [
        ("self", module),
        ("self.sub1", _module_attrs(module).sub1),
        ("self.sub2", _module_attrs(module).sub2),
    ]
    assert sorted(module.named_sub_modules()) == sorted(expected)


def test_non_base_module_attributes():
    module = Module()
    _module_attrs(module).sub = Module()
    _module_attrs(module).not_a_sub = "Not a self"
    expected = [("self", module), ("self.sub", _module_attrs(module).sub)]
    assert list(module.named_sub_modules()) == expected


def test_complex_module_traversal():
    root = Module()
    root_attrs = _module_attrs(root)
    sub_attrs = _module_attrs(Module())
    root_attrs.sub_module = sub_attrs
    sub_attrs.nested_list = [Module(), {"key": Module()}]
    sub_attrs.nested_tuple = (Module(), [Module(), Module()])
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
    root_attrs = _module_attrs(root)
    sub_attrs = _module_attrs(Module())
    root_attrs.sub_module = sub_attrs
    sub_attrs.nested_list = [Module(), {"key": Module()}]
    same_module = Module()
    sub_attrs.nested_tuple = (Module(), [same_module, same_module])
    expected_names = {
        "self",
        "self.sub_module",
        "self.sub_module.nested_list[0]",
        "self.sub_module.nested_list[1][key]",
        "self.sub_module.nested_tuple[0]",
        "self.sub_module.nested_tuple[1][0]",
    }
    found_names = {name for name, _ in root.named_sub_modules()}
    assert found_names == expected_names, (
        f"Missing or extra modules found. Missing: {expected_names - found_names}, Extra: {found_names - expected_names}"
    )


def test_named_parameters_traverses_nested_containers():
    root = Module()
    root_attrs = _module_attrs(root)
    sub_attrs = _module_attrs(Module())
    root_attrs.sub_module = sub_attrs
    sub_attrs.nested_predict = Predict(ts("question -> answer", instructions="Answer the question."))
    sub_attrs.nested_list = [Predict(ts("question -> answer", instructions="Answer the question."))]
    sub_attrs.nested_dict = {"key": Predict(ts("question -> answer", instructions="Answer the question."))}
    found_names = {name for name, _ in root.named_parameters()}
    assert "self.sub_module.nested_predict" in found_names
    assert "self.sub_module.nested_list[0]" in found_names
    assert "self.sub_module.nested_dict[key]" in found_names


def test_complex_module_set_attribute_by_name():
    root = Module()
    root_attrs = _module_attrs(root)
    sub_attrs = _module_attrs(Module())
    root_attrs.sub_module = sub_attrs
    sub_attrs.nested_list = [Module(), {"key": Module()}]
    same_module = Module()
    sub_attrs.nested_tuple = (Module(), [same_module, same_module])
    set_attribute_by_name(root, "test_attrib", True)
    assert _module_attrs(root).test_attrib is True
    set_attribute_by_name(root, "sub_module.test_attrib", True)
    sub_attrs = _module_attrs(root_attrs.sub_module)
    assert sub_attrs.test_attrib is True
    set_attribute_by_name(root, "sub_module.nested_list[0].test_attrib", True)
    assert _module_attrs(sub_attrs.nested_list[0]).test_attrib is True
    set_attribute_by_name(root, "sub_module.nested_list[1]['key'].test_attrib", True)
    nested_key_module = cast("Module", sub_attrs.nested_list[1]["key"])
    assert _module_attrs(nested_key_module).test_attrib is True
    set_attribute_by_name(root, "sub_module.nested_tuple[0].test_attrib", True)
    assert _module_attrs(sub_attrs.nested_tuple[0]).test_attrib is True
    set_attribute_by_name(root, "sub_module.nested_tuple[1][0].test_attrib", True)
    assert _module_attrs(sub_attrs.nested_tuple[1][0]).test_attrib is True
    assert _module_attrs(sub_attrs.nested_tuple[1][1]).test_attrib is True


class DuplicateModule(Module):
    def __init__(self):
        super().__init__()
        self.p0 = Predict(ts("question -> answer", instructions="Answer the question."))
        self.p1 = self.p0


def test_named_parameters_duplicate_references():
    module = DuplicateModule()
    module.named_parameters()


def test_load_state_is_transactional():
    Sig = ts("question -> answer")

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
