import pydantic
import pytest

from dspy.predict.predict import Predict
from dspy.primitives.module import Module
from dspy.signatures.signature import Signature


def test_basic_custom_type_resolution():
    """Test basic custom type resolution with both explicit and automatic mapping."""

    class CustomType(pydantic.BaseModel):
        value: str

    # Custom types can be explicitly mapped
    explicit_sig = Signature(
        "input: CustomType -> output: str",  # ty:ignore[too-many-positional-arguments]
        custom_types={"CustomType": CustomType},  # ty:ignore[unknown-argument]
    )
    assert explicit_sig.input_fields["input"].annotation == CustomType  # ty:ignore[unresolved-attribute]

    # Custom types can also be auto-resolved from caller's scope
    auto_sig = Signature("input: CustomType -> output: str")  # ty:ignore[too-many-positional-arguments]
    assert auto_sig.input_fields["input"].annotation == CustomType  # ty:ignore[unresolved-attribute]


def test_type_alias_for_nested_types():
    """Test using type aliases for nested types."""

    class Container:
        class NestedType(pydantic.BaseModel):
            value: str

    NestedType = Container.NestedType
    assert NestedType is Container.NestedType
    alias_sig = Signature("input: str -> output: NestedType")  # ty:ignore[too-many-positional-arguments]
    assert alias_sig.output_fields["output"].annotation == Container.NestedType  # ty:ignore[unresolved-attribute]

    class Container2:
        class Query(pydantic.BaseModel):
            text: str

        class Score(pydantic.BaseModel):
            score: float

    signature = Signature("query: Container2.Query -> score: Container2.Score")  # ty:ignore[too-many-positional-arguments]
    assert signature.output_fields["score"].annotation == Container2.Score  # ty:ignore[unresolved-attribute]


class GlobalCustomType(pydantic.BaseModel):
    """A type defined at module level for testing module-level resolution."""

    value: str
    notes: str = ""


def test_module_level_type_resolution():
    """Test resolution of types defined at module level."""
    # Module-level types can be auto-resolved
    sig = Signature("name: str -> result: GlobalCustomType")  # ty:ignore[too-many-positional-arguments]
    assert sig.output_fields["result"].annotation == GlobalCustomType  # ty:ignore[unresolved-attribute]


# Create module-level nested class for testing
class OuterContainer:
    class InnerType(pydantic.BaseModel):
        name: str
        value: int


def test_recommended_patterns():
    """Test recommended patterns for working with custom types in signatures."""

    # PATTERN 1: Local type with auto-resolution
    class LocalType(pydantic.BaseModel):
        value: str

    sig1 = Signature("input: str -> output: LocalType")  # ty:ignore[too-many-positional-arguments]
    assert sig1.output_fields["output"].annotation == LocalType  # ty:ignore[unresolved-attribute]

    # PATTERN 2: Module-level type with auto-resolution
    sig2 = Signature("input: str -> output: GlobalCustomType")  # ty:ignore[too-many-positional-arguments]
    assert sig2.output_fields["output"].annotation == GlobalCustomType  # ty:ignore[unresolved-attribute]

    # PATTERN 3: Nested type with dot notation
    sig3 = Signature("input: str -> output: OuterContainer.InnerType")  # ty:ignore[too-many-positional-arguments]
    assert sig3.output_fields["output"].annotation == OuterContainer.InnerType  # ty:ignore[unresolved-attribute]

    # PATTERN 4: Nested type using alias
    InnerTypeAlias = OuterContainer.InnerType
    sig4 = Signature("input: str -> output: InnerTypeAlias")  # ty:ignore[too-many-positional-arguments]
    assert sig4.output_fields["output"].annotation == InnerTypeAlias  # ty:ignore[unresolved-attribute]

    # PATTERN 5: Nested type with dot notation
    sig5 = Signature("input: str -> output: OuterContainer.InnerType")  # ty:ignore[too-many-positional-arguments]
    assert sig5.output_fields["output"].annotation == OuterContainer.InnerType  # ty:ignore[unresolved-attribute]


def test_expected_failure():
    # InnerType DNE when not OuterContainer.InnerTypes, so this type shouldnt be resolved
    with pytest.raises(ValueError):  # noqa: PT011
        Signature("input: str -> output: InnerType")  # ty:ignore[too-many-positional-arguments]


def test_module_type_resolution():
    class TestModule(Module):
        def __init__(self):
            super().__init__()
            self.predict = Predict("input: str -> output: OuterContainer.InnerType")  # ty:ignore[invalid-assignment]

        def predict(self, input: str) -> str:
            return input

    module = TestModule()
    sig = module.predict.signature  # ty:ignore[unresolved-attribute]
    assert sig.output_fields["output"].annotation == OuterContainer.InnerType


def test_basic_custom_type_resolution():  # noqa: F811
    class CustomType(pydantic.BaseModel):
        value: str

    sig = Signature("input: CustomType -> output: str", custom_types={"CustomType": CustomType})  # ty:ignore[too-many-positional-arguments, unknown-argument]
    assert sig.input_fields["input"].annotation == CustomType  # ty:ignore[unresolved-attribute]

    sig = Signature("input: CustomType -> output: str")  # ty:ignore[too-many-positional-arguments]
    assert sig.input_fields["input"].annotation == CustomType  # ty:ignore[unresolved-attribute]
