import pytest

from dspy.history import TurnLog
from dspy.adapters.types.image import Image
from dspy.primitives.example import Example


def test_example_initialization():
    example = Example.from_record({"a": 1, "b": 2})
    assert example.a == 1
    assert example.b == 2


def test_example_initialization_from_base():
    base = Example.from_record({"a": 1, "b": 2})
    example = base.fork(c=3)
    assert example.a == 1
    assert example.b == 2
    assert example.c == 3


def test_example_initialization_from_dict():
    base_dict = {"a": 1, "b": 2}
    example = Example.from_record({**base_dict, "c": 3})
    assert example.a == 1
    assert example.b == 2
    assert example.c == 3


def test_example_set_get_item():
    example = Example.from_record({})
    example["a"] = 1
    assert example["a"] == 1


def test_example_attribute_access():
    example = Example.from_record({"a": 1})
    assert example.a == 1
    example.a = 2
    assert example.a == 2


def test_example_deletion():
    example = Example.from_record({"a": 1, "b": 2})
    del example["a"]
    with pytest.raises(AttributeError):
        _ = example.a


def test_example_len():
    example = Example.from_record({"a": 1, "b": 2, "dspy_hidden": 3})
    assert len(example) == 2


def test_example_repr_str_img():
    example = Example.from_record(
        {"img": Image(url="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7")}
    )
    assert (
        repr(example)
        == "Example({'img': Image(url=data:image/gif;base64,<IMAGE_BASE_64_ENCODED(56)>)}) (input_keys=[])"
    )
    assert (
        str(example) == "Example({'img': Image(url=data:image/gif;base64,<IMAGE_BASE_64_ENCODED(56)>)}) (input_keys=[])"
    )


def test_example_repr_str():
    example = Example.from_record({"a": 1})
    assert repr(example) == "Example({'a': 1}) (input_keys=[])"
    assert str(example) == "Example({'a': 1}) (input_keys=[])"


def test_example_eq():
    example1 = Example.from_record({"a": 1, "b": 2})
    example2 = Example.from_record({"a": 1, "b": 2})
    assert example1 == example2
    assert example1 != ""


def test_example_hash():
    example1 = Example.from_record({"a": 1, "b": 2})
    example2 = Example.from_record({"a": 1, "b": 2})
    assert hash(example1) == hash(example2)


def test_example_keys_values_items():
    example = Example.from_record({"a": 1, "b": 2, "dspy_hidden": 3})
    assert set(example.keys()) == {"a", "b"}
    assert 1 in example.values()
    assert ("b", 2) in example.items()


def test_example_get():
    example = Example.from_record({"a": 1, "b": 2})
    assert example.get("a") == 1
    assert example.get("c", "default") == "default"


def test_example_with_inputs():
    example = Example.from_record({"a": 1, "b": 2}, input_keys=("a"))
    assert example._input_keys == {"a"}


def test_example_inputs_labels():
    example = Example.from_record({"a": 1, "b": 2}, input_keys=("a"))
    inputs = example.as_inputs()
    assert inputs == {"a": 1}
    labels = example.as_labels()
    assert labels == {"b": 2}


def test_example_copy_without():
    example = Example.from_record({"a": 1, "b": 2})
    copied = example.fork(c=3)
    assert copied.a == 1
    assert copied.c == 3
    without_a = copied.without("a")
    with pytest.raises(AttributeError):
        _ = without_a.a


def test_example_to_dict():
    example = Example.from_record({"a": 1, "b": 2})
    assert example.to_dict() == {"a": 1, "b": 2}


def test_example_to_dict_with_history():
    history = TurnLog(turns=(
            {"question": "What is the capital of France?", "answer": "Paris"},
            {"question": "What is the capital of Germany?", "answer": "Berlin"},
        ))
    example = Example.from_record({"question": "Test question", "history": history, "answer": "Test answer"})
    result = example.to_dict()
    assert isinstance(result, dict)
    assert "history" in result
    assert isinstance(result["history"], dict)
    assert "turns" in result["history"]
    assert result["history"]["turns"] == (
        {"question": "What is the capital of France?", "answer": "Paris"},
        {"question": "What is the capital of Germany?", "answer": "Berlin"},
    )
    import json

    json_str = json.dumps(result)
    restored = json.loads(json_str)
    assert list(restored["history"]["turns"]) == list(result["history"]["turns"])
