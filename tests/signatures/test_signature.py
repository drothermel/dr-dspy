import pickle
from types import UnionType
from typing import Any, Union

import cloudpickle
import pydantic
import pytest

from dspy.dsp.utils.settings import settings
from dspy.predict.predict import Predict
from dspy.signatures.field import InputField, OutputField
from dspy.signatures.signature import Signature, infer_prefix
from dspy.utils.dummies import DummyLM


def _is_union(annotation: object) -> bool:
    return isinstance(annotation, UnionType)


def test_field_types_and_custom_attributes():
    class TestSignature(Signature):
        """Instructions"""

        input1: str = InputField()
        input2: int = InputField()
        output1: list[str] = OutputField()
        output2 = OutputField()

    assert TestSignature.instructions == "Instructions"
    assert TestSignature.input_fields["input1"].annotation == str
    assert TestSignature.input_fields["input2"].annotation == int
    assert TestSignature.output_fields["output1"].annotation == list[str]
    assert TestSignature.output_fields["output2"].annotation == str


def test_no_input_output():
    with pytest.raises(TypeError):

        class TestSignature(Signature):
            input1: str


def test_no_input_output2():
    with pytest.raises(TypeError):

        class TestSignature(Signature):
            input1: str = pydantic.Field()


def test_all_fields_have_prefix():
    class TestSignature(Signature):
        input = InputField(prefix="Modified:")
        output = OutputField()

    assert TestSignature.input_fields["input"].json_schema_extra["prefix"] == "Modified:"  # ty:ignore[not-subscriptable]
    assert TestSignature.output_fields["output"].json_schema_extra["prefix"] == "Output:"  # ty:ignore[not-subscriptable]


def test_signature_parsing():
    signature = Signature("input1, input2 -> output")  # ty:ignore[too-many-positional-arguments]
    assert "input1" in signature.input_fields  # ty:ignore[unresolved-attribute]
    assert "input2" in signature.input_fields  # ty:ignore[unresolved-attribute]
    assert "output" in signature.output_fields  # ty:ignore[unresolved-attribute]


def test_duplicate_input_output_field_names_raise():
    with pytest.raises(ValueError, match="distinct names"):
        Signature("value -> value")  # ty:ignore[too-many-positional-arguments]


def test_with_signature():
    signature1 = Signature("input1, input2 -> output")  # ty:ignore[too-many-positional-arguments]
    signature2 = signature1.with_instructions("This is a test")
    assert signature2.instructions == "This is a test"
    assert signature1 is not signature2, "The type should be immutable"


def test_with_updated_field():
    signature1 = Signature("input1, input2 -> output")  # ty:ignore[too-many-positional-arguments]
    signature2 = signature1.with_updated_fields("input1", prefix="Modified:")
    assert signature2.input_fields["input1"].json_schema_extra["prefix"] == "Modified:"  # ty:ignore[not-subscriptable]
    assert signature1.input_fields["input1"].json_schema_extra["prefix"] == "Input 1:"  # ty:ignore[unresolved-attribute]
    assert signature1 is not signature2, "The type should be immutable"
    for key in signature1.fields:  # ty:ignore[unresolved-attribute]
        if key != "input1":
            assert signature1.fields[key].json_schema_extra == signature2.fields[key].json_schema_extra  # ty:ignore[unresolved-attribute]
    assert signature1.instructions == signature2.instructions  # ty:ignore[unresolved-attribute]


def test_empty_signature():
    with pytest.raises(ValueError):  # noqa: PT011
        Signature("")  # ty:ignore[too-many-positional-arguments]


def test_instructions_signature():
    with pytest.raises(ValueError):  # noqa: PT011
        Signature("")  # ty:ignore[too-many-positional-arguments]


def test_signature_instructions():
    sig1 = Signature("input1 -> output1", instructions="This is a test")  # ty:ignore[too-many-positional-arguments, unknown-argument]
    assert sig1.instructions == "This is a test"  # ty:ignore[unresolved-attribute]
    sig2 = Signature("input1 -> output1", "This is a test")  # ty:ignore[too-many-positional-arguments]
    assert sig2.instructions == "This is a test"  # ty:ignore[unresolved-attribute]


def test_signature_instructions_none():
    sig1 = Signature("a, b -> c")  # ty:ignore[too-many-positional-arguments]
    assert sig1.instructions == "Given the fields `a`, `b`, produce the fields `c`."  # ty:ignore[unresolved-attribute]


def test_signature_from_dict():
    signature = Signature({"input1": InputField(), "input2": InputField(), "output": OutputField()})  # ty:ignore[too-many-positional-arguments]
    for k in ["input1", "input2", "output"]:
        assert k in signature.fields  # ty:ignore[unresolved-attribute]
        assert signature.fields[k].annotation == str  # ty:ignore[unresolved-attribute]


def test_signature_equality():
    sig1 = Signature("input1 -> output1")  # ty:ignore[too-many-positional-arguments]
    sig2 = Signature("input1 -> output1")  # ty:ignore[too-many-positional-arguments]
    assert sig1.equals(sig2)


def test_signature_inequality():
    sig1 = Signature("input1 -> output1")  # ty:ignore[too-many-positional-arguments]
    sig2 = Signature("input2 -> output2")  # ty:ignore[too-many-positional-arguments]
    assert not sig1.equals(sig2)


def test_equality_format():
    class TestSignature(Signature):
        input = InputField(format=lambda x: x)
        output = OutputField()

    assert TestSignature.equals(TestSignature)


def test_signature_reverse():
    sig = Signature("input1 -> output1")  # ty:ignore[too-many-positional-arguments]
    assert sig.signature == "input1 -> output1"  # ty:ignore[unresolved-attribute]


def test_insert_field_at_various_positions():
    class InitialSignature(Signature):
        input1: str = InputField()
        output1: int = OutputField()

    s1 = InitialSignature.prepend("new_input_start", InputField(), str)
    s2 = InitialSignature.append("new_input_end", InputField(), str)
    assert list(s1.input_fields.keys())[0] == "new_input_start"  # noqa: RUF015
    assert list(s2.input_fields.keys())[-1] == "new_input_end"

    s3 = InitialSignature.prepend("new_output_start", OutputField(), str)
    s4 = InitialSignature.append("new_output_end", OutputField(), str)
    assert list(s3.output_fields.keys())[0] == "new_output_start"  # noqa: RUF015
    assert list(s4.output_fields.keys())[-1] == "new_output_end"


def test_order_preserved_with_mixed_annotations():
    class ExampleSignature(Signature):
        text: str = InputField()
        output = OutputField()
        pass_evaluation: bool = OutputField()

    expected_order = ["text", "output", "pass_evaluation"]
    actual_order = list(ExampleSignature.fields.keys())
    assert actual_order == expected_order


def test_infer_prefix():
    assert infer_prefix("someAttributeName42IsCool") == "Some Attribute Name 42 Is Cool"
    assert infer_prefix("version2Update") == "Version 2 Update"
    assert infer_prefix("modelT45Enhanced") == "Model T 45 Enhanced"
    assert infer_prefix("someAttributeName") == "Some Attribute Name"
    assert infer_prefix("some_attribute_name") == "Some Attribute Name"
    assert infer_prefix("URLAddress") == "URL Address"
    assert infer_prefix("isHTTPSecure") == "Is HTTP Secure"
    assert infer_prefix("isHTTPSSecure123") == "Is HTTPS Secure 123"


def test_insantiating():
    sig = Signature("input -> output")  # ty:ignore[too-many-positional-arguments]
    assert issubclass(sig, Signature)  # ty:ignore[invalid-argument-type]
    assert sig.__name__ == "StringSignature"
    value = sig(input="test", output="test")
    assert isinstance(value, sig)


def test_insantiating2():
    class SubSignature(Signature):
        input = InputField()
        output = OutputField()

    assert issubclass(SubSignature, Signature)
    assert SubSignature.__name__ == "SubSignature"
    value = SubSignature(input="test", output="test")  # ty:ignore[unknown-argument]
    assert isinstance(value, SubSignature)


def test_multiline_instructions():
    lm = DummyLM([{"output": "short answer"}])
    settings.configure(lm=lm)

    class MySignature(Signature):
        """First line
        Second line
            Third line"""

        output = OutputField()

    predictor = Predict(MySignature)
    assert predictor().output == "short answer"


def test_dump_and_load_state():
    class CustomSignature(Signature):
        """I am just an instruction."""

        sentence = InputField(desc="I am an innocent input!")
        sentiment = OutputField()

    state = CustomSignature.dump_state()
    expected = {
        "instructions": "I am just an instruction.",
        "fields": [
            {
                "prefix": "Sentence:",
                "description": "I am an innocent input!",
            },
            {
                "prefix": "Sentiment:",
                "description": "${sentiment}",
            },
        ],
    }
    assert state == expected

    class CustomSignature2(Signature):
        """I am a malicious instruction."""

        sentence = InputField(desc="I am an malicious input!")
        sentiment = OutputField()

    assert CustomSignature2.dump_state() != expected
    # Overwrite the state with the state of CustomSignature.
    loaded_signature = CustomSignature2.load_state(state)
    assert loaded_signature.instructions == "I am just an instruction."
    # After `load_state`, the state should be the same as CustomSignature.
    assert loaded_signature.dump_state() == expected
    # CustomSignature2 should not have been modified.
    assert CustomSignature2.instructions == "I am a malicious instruction."
    assert CustomSignature2.fields["sentence"].json_schema_extra["desc"] == "I am an malicious input!"  # ty:ignore[not-subscriptable]
    assert CustomSignature2.fields["sentiment"].json_schema_extra["prefix"] == "Sentiment:"  # ty:ignore[not-subscriptable]


def test_typed_signatures_basic_types():
    sig = Signature("input1: int, input2: str -> output: float")  # ty:ignore[too-many-positional-arguments]
    assert "input1" in sig.input_fields  # ty:ignore[unresolved-attribute]
    assert sig.input_fields["input1"].annotation == int  # ty:ignore[unresolved-attribute]
    assert "input2" in sig.input_fields  # ty:ignore[unresolved-attribute]
    assert sig.input_fields["input2"].annotation == str  # ty:ignore[unresolved-attribute]
    assert "output" in sig.output_fields  # ty:ignore[unresolved-attribute]
    assert sig.output_fields["output"].annotation == float  # ty:ignore[unresolved-attribute]


def test_typed_signatures_generics():
    sig = Signature("input_list: list[int], input_dict: dict[str, float] -> output_tuple: tuple[str, int]")  # ty:ignore[too-many-positional-arguments]
    assert "input_list" in sig.input_fields  # ty:ignore[unresolved-attribute]
    assert sig.input_fields["input_list"].annotation == list[int]  # ty:ignore[unresolved-attribute]
    assert "input_dict" in sig.input_fields  # ty:ignore[unresolved-attribute]
    assert sig.input_fields["input_dict"].annotation == dict[str, float]  # ty:ignore[unresolved-attribute]
    assert "output_tuple" in sig.output_fields  # ty:ignore[unresolved-attribute]
    assert sig.output_fields["output_tuple"].annotation == tuple[str, int]  # ty:ignore[unresolved-attribute]


def test_typed_signatures_unions_and_optionals():
    sig = Signature("input_opt: str | None, input_union: int | None -> output_union: int | str")  # ty:ignore[too-many-positional-arguments]
    assert "input_opt" in sig.input_fields  # ty:ignore[unresolved-attribute]
    input_opt_annotation = sig.input_fields["input_opt"].annotation  # ty:ignore[unresolved-attribute]
    assert _is_union(input_opt_annotation)
    assert str in input_opt_annotation.__args__
    assert type(None) in input_opt_annotation.__args__

    assert "input_union" in sig.input_fields  # ty:ignore[unresolved-attribute]
    input_union_annotation = sig.input_fields["input_union"].annotation  # ty:ignore[unresolved-attribute]
    assert _is_union(input_union_annotation)
    assert int in input_union_annotation.__args__
    assert type(None) in input_union_annotation.__args__

    assert "output_union" in sig.output_fields  # ty:ignore[unresolved-attribute]
    output_union_annotation = sig.output_fields["output_union"].annotation  # ty:ignore[unresolved-attribute]
    assert _is_union(output_union_annotation)
    assert int in output_union_annotation.__args__
    assert str in output_union_annotation.__args__


def test_typed_signatures_any():
    sig = Signature("input_any: Any -> output_any: Any")  # ty:ignore[too-many-positional-arguments]
    assert "input_any" in sig.input_fields  # ty:ignore[unresolved-attribute]
    assert sig.input_fields["input_any"].annotation == Any  # ty:ignore[unresolved-attribute]
    assert "output_any" in sig.output_fields  # ty:ignore[unresolved-attribute]
    assert sig.output_fields["output_any"].annotation == Any  # ty:ignore[unresolved-attribute]


def test_typed_signatures_nested():
    sig = Signature("input_nested: list[str | int] -> output_nested: tuple[int, float | None, list[str]]")  # ty:ignore[too-many-positional-arguments]
    input_nested_ann = sig.input_fields["input_nested"].annotation  # ty:ignore[unresolved-attribute]
    assert getattr(input_nested_ann, "__origin__", None) is list
    assert len(input_nested_ann.__args__) == 1
    union_arg = input_nested_ann.__args__[0]
    assert _is_union(union_arg)
    assert str in union_arg.__args__
    assert int in union_arg.__args__

    output_nested_ann = sig.output_fields["output_nested"].annotation  # ty:ignore[unresolved-attribute]
    assert getattr(output_nested_ann, "__origin__", None) is tuple
    assert output_nested_ann.__args__[0] == int
    second_arg = output_nested_ann.__args__[1]
    assert _is_union(second_arg)
    assert float in second_arg.__args__
    assert type(None) in second_arg.__args__
    # The third arg is list[str]
    third_arg = output_nested_ann.__args__[2]
    assert getattr(third_arg, "__origin__", None) is list
    assert third_arg.__args__[0] == str


def test_typed_signatures_from_dict():
    fields = {
        "input_str_list": (list[str], InputField()),
        "input_dict_int": (dict[str, int], InputField()),
        "output_tup": (tuple[int, float], OutputField()),
    }
    sig = Signature(fields)  # ty:ignore[too-many-positional-arguments]
    assert "input_str_list" in sig.input_fields  # ty:ignore[unresolved-attribute]
    assert sig.input_fields["input_str_list"].annotation == list[str]  # ty:ignore[unresolved-attribute]
    assert "input_dict_int" in sig.input_fields  # ty:ignore[unresolved-attribute]
    assert sig.input_fields["input_dict_int"].annotation == dict[str, int]  # ty:ignore[unresolved-attribute]
    assert "output_tup" in sig.output_fields  # ty:ignore[unresolved-attribute]
    assert sig.output_fields["output_tup"].annotation == tuple[int, float]  # ty:ignore[unresolved-attribute]


def test_typed_signatures_complex_combinations():
    sig = Signature(
        "input_complex: dict[str, list[tuple[int, str] | None]] -> output_complex: list[str] | dict[str, Any]"  # ty:ignore[too-many-positional-arguments]
    )
    input_complex_ann = sig.input_fields["input_complex"].annotation  # ty:ignore[unresolved-attribute]
    assert getattr(input_complex_ann, "__origin__", None) is dict
    key_arg, value_arg = input_complex_ann.__args__
    assert key_arg == str
    assert getattr(value_arg, "__origin__", None) is list
    inner_union = value_arg.__args__[0]
    assert _is_union(inner_union)
    tuple_type = [t for t in inner_union.__args__ if t != type(None)][0]  # noqa: RUF015
    assert getattr(tuple_type, "__origin__", None) is tuple
    assert tuple_type.__args__ == (int, str)

    output_complex_ann = sig.output_fields["output_complex"].annotation  # ty:ignore[unresolved-attribute]
    assert _is_union(output_complex_ann)
    assert len(output_complex_ann.__args__) == 2
    possible_args = set(output_complex_ann.__args__)
    # Expecting list[str] and dict[str, Any]
    # Because sets don't preserve order, just check membership.
    # Find the list[str] arg
    list_arg = next(a for a in possible_args if getattr(a, "__origin__", None) is list)
    dict_arg = next(a for a in possible_args if getattr(a, "__origin__", None) is dict)
    assert list_arg.__args__ == (str,)
    k, v = dict_arg.__args__
    assert k == str
    assert v == Any


def test_make_signature_from_string():
    sig = Signature("input1: int, input2: dict[str, int] -> output1: list[str], output2: int | str")  # ty:ignore[too-many-positional-arguments]
    assert "input1" in sig.input_fields  # ty:ignore[unresolved-attribute]
    assert sig.input_fields["input1"].annotation == int  # ty:ignore[unresolved-attribute]
    assert "input2" in sig.input_fields  # ty:ignore[unresolved-attribute]
    assert sig.input_fields["input2"].annotation == dict[str, int]  # ty:ignore[unresolved-attribute]
    assert "output1" in sig.output_fields  # ty:ignore[unresolved-attribute]
    assert sig.output_fields["output1"].annotation == list[str]  # ty:ignore[unresolved-attribute]
    assert "output2" in sig.output_fields  # ty:ignore[unresolved-attribute]
    assert _is_union(sig.output_fields["output2"].annotation)  # ty:ignore[unresolved-attribute]
    assert set(sig.output_fields["output2"].annotation.__args__) == {int, str}  # ty:ignore[unresolved-attribute]


def test_signature_field_with_constraints():
    class MySignature(Signature):
        inputs: str = InputField()
        outputs1: str = OutputField(min_length=5, max_length=10)
        outputs2: int = OutputField(ge=5, le=10)

    assert "outputs1" in MySignature.output_fields
    output1_constraints = MySignature.output_fields["outputs1"].json_schema_extra["constraints"]  # ty:ignore[not-subscriptable]

    assert "minimum length: 5" in output1_constraints
    assert "maximum length: 10" in output1_constraints

    assert "outputs2" in MySignature.output_fields
    output2_constraints = MySignature.output_fields["outputs2"].json_schema_extra["constraints"]  # ty:ignore[not-subscriptable]
    assert "greater than or equal to: 5" in output2_constraints
    assert "less than or equal to: 10" in output2_constraints


def test_basic_custom_type():
    class CustomType(pydantic.BaseModel):
        value: str

    test_signature = Signature(
        "input: CustomType -> output: str",  # ty:ignore[too-many-positional-arguments]
        custom_types={"CustomType": CustomType},  # ty:ignore[unknown-argument]
    )

    assert test_signature.input_fields["input"].annotation == CustomType  # ty:ignore[unresolved-attribute]

    lm = DummyLM([{"output": "processed"}])
    settings.configure(lm=lm)

    custom_obj = CustomType(value="test")
    pred = Predict(test_signature)(input=custom_obj)  # ty:ignore[invalid-argument-type]
    assert pred.output == "processed"


def test_custom_type_from_different_module():
    from pathlib import Path

    test_signature = Signature("input: Path -> output: str")  # ty:ignore[too-many-positional-arguments]
    assert test_signature.input_fields["input"].annotation == Path  # ty:ignore[unresolved-attribute]

    lm = DummyLM([{"output": "/test/path"}])
    settings.configure(lm=lm)

    path_obj = Path("/test/path")
    pred = Predict(test_signature)(input=path_obj)  # ty:ignore[invalid-argument-type]
    assert pred.output == "/test/path"


def test_pep604_union_type_inline():
    sig = Signature(
        "input1: str | None, input2: None | int -> output_union: int | str"  # ty:ignore[too-many-positional-arguments]
    )

    assert "input1" in sig.input_fields  # ty:ignore[unresolved-attribute]
    input1_annotation = sig.input_fields["input1"].annotation  # ty:ignore[unresolved-attribute]
    assert _is_union(input1_annotation)
    assert str in input1_annotation.__args__
    assert type(None) in input1_annotation.__args__

    assert "input2" in sig.input_fields  # ty:ignore[unresolved-attribute]
    input2_annotation = sig.input_fields["input2"].annotation  # ty:ignore[unresolved-attribute]
    assert _is_union(input2_annotation)
    assert int in input2_annotation.__args__
    assert type(None) in input2_annotation.__args__

    assert "output_union" in sig.output_fields  # ty:ignore[unresolved-attribute]
    output_union_annotation = sig.output_fields["output_union"].annotation  # ty:ignore[unresolved-attribute]
    assert _is_union(output_union_annotation)
    assert int in output_union_annotation.__args__
    assert str in output_union_annotation.__args__


def test_reject_legacy_union_string_signatures():
    with pytest.raises(ValueError, match=r"typing\.Optional"):
        Signature("input: Optional[str] -> output: str")  # ty:ignore[too-many-positional-arguments]

    with pytest.raises(ValueError, match=r"typing\.Union"):
        Signature("input: int -> output: Union[int, str]")  # ty:ignore[too-many-positional-arguments]


def test_reject_legacy_union_class_signatures():
    with pytest.raises(ValueError, match="typing.Union"):

        class LegacyUnionSignature(Signature):
            input: str = InputField()
            output: Union[int, str] = OutputField()


def test_pep604_union_type_inline_nested():
    sig = Signature(
        "input: str | (int | float) | None -> output: str"  # ty:ignore[too-many-positional-arguments]
    )
    assert "input" in sig.input_fields  # ty:ignore[unresolved-attribute]
    input_annotation = sig.input_fields["input"].annotation  # ty:ignore[unresolved-attribute]

    assert _is_union(input_annotation)
    assert set(input_annotation.__args__) == {str, int, float, type(None)}


def test_pep604_union_type_class_nested():
    class Sig1(Signature):
        input: str | (int | float) | None = InputField()
        output: str = OutputField()

    assert "input" in Sig1.input_fields
    input_annotation = Sig1.input_fields["input"].annotation

    assert isinstance(input_annotation, UnionType)
    assert set(input_annotation.__args__) == {str, int, float, type(None)}


def test_pep604_union_type_insert():
    class PEP604Signature(Signature):
        input: str | None = InputField()
        output: int | str = OutputField()

    # This test ensures that inserting a field into a signature with a PEP 604 UnionType works

    # Insert a new input field at the start
    NewSig = PEP604Signature.prepend("new_input", InputField(), float | int)
    assert "new_input" in NewSig.input_fields

    new_input_annotation = NewSig.input_fields["new_input"].annotation
    assert isinstance(new_input_annotation, UnionType)
    assert set(new_input_annotation.__args__) == {float, int}

    # The original union type field should still be present and correct
    input_annotation = NewSig.input_fields["input"].annotation
    output_annotation = NewSig.output_fields["output"].annotation

    assert isinstance(input_annotation, UnionType)
    assert str in input_annotation.__args__
    assert type(None) in input_annotation.__args__

    assert isinstance(output_annotation, UnionType)
    assert set(output_annotation.__args__) == {int, str}


def test_pep604_union_type_with_custom_types():
    class CustomType(pydantic.BaseModel):
        value: str

    sig = Signature(
        "input: CustomType | None -> output: int | str",  # ty:ignore[too-many-positional-arguments]
        custom_types={"CustomType": CustomType},  # ty:ignore[unknown-argument]
    )

    input_annotation = sig.input_fields["input"].annotation  # ty:ignore[unresolved-attribute]
    assert _is_union(input_annotation)
    assert CustomType in input_annotation.__args__
    assert type(None) in input_annotation.__args__

    output_annotation = sig.output_fields["output"].annotation  # ty:ignore[unresolved-attribute]
    assert _is_union(output_annotation)
    assert set(output_annotation.__args__) == {int, str}

    lm = DummyLM([{"output": "processed"}])
    settings.configure(lm=lm)

    custom_obj = CustomType(value="test")
    pred = Predict(sig)(input=custom_obj)  # ty:ignore[invalid-argument-type]
    assert pred.output == "processed"


def test_signature_cloudpickle_roundtrip():
    class MySignature(Signature):
        """Answer the question."""

        context: list[str] = InputField()
        question: str = InputField()
        answer: str = OutputField()

    data = cloudpickle.dumps(MySignature)
    loaded = pickle.loads(data)  # noqa: S301

    assert loaded.__name__ == "MySignature"
    assert list(loaded.input_fields.keys()) == ["context", "question"]
    assert list(loaded.output_fields.keys()) == ["answer"]
    assert loaded.instructions == "Answer the question."


def test_predict_cloudpickle_roundtrip():
    class QA(Signature):
        """Answer the question."""

        question: str = InputField()
        answer: str = OutputField()

    predict = Predict(QA)
    data = cloudpickle.dumps(predict)
    loaded = pickle.loads(data)  # noqa: S301

    assert list(loaded.signature.fields.keys()) == ["question", "answer"]
