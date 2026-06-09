from typing import Literal

import pytest

from dspy.adapters.types.image import Image
from dspy.task_spec import (
    FieldBinding,
    field_bindings,
    input_field,
    make_task_spec,
    output_field,
    validate_task_inputs_from_spec,
)
from dspy.task_spec.field_spec import FieldRole


def test_field_bindings_returns_named_specs():
    spec = make_task_spec("q -> a", instructions="Answer.")
    bindings = field_bindings(spec, role=FieldRole.INPUT)
    assert bindings == (FieldBinding(name="q", field=spec.input_fields["q"]),)


def test_validate_task_inputs_from_spec_applies_defaults():
    spec = make_task_spec(
        inputs=[input_field("q", desc="Question.", default="default-q")],
        outputs=[output_field("a", desc="Answer.")],
        instructions="Answer.",
    )
    validated = validate_task_inputs_from_spec(spec, {})
    assert validated == {"q": "default-q"}


def test_validate_task_inputs_from_spec_rejects_unknown_keys():
    spec = make_task_spec("q -> a", instructions="Answer.")
    with pytest.raises(ValueError, match="Unknown task input field"):
        validate_task_inputs_from_spec(spec, {"extra": 1})


def test_validate_task_inputs_from_spec_rejects_none_for_required_str():
    spec = make_task_spec(
        inputs=[input_field("q", str, desc="The question.")],
        outputs=[output_field("a", desc="The answer.")],
        instructions="Answer.",
    )
    with pytest.raises(ValueError, match="got incompatible value None"):
        validate_task_inputs_from_spec(spec, {"q": None})


def test_validate_task_inputs_from_spec_accepts_none_for_optional_str():
    spec = make_task_spec(
        inputs=[input_field("q", str | None, desc="Optional question.")],
        outputs=[output_field("a", desc="The answer.")],
        instructions="Answer.",
    )
    validated = validate_task_inputs_from_spec(spec, {"q": None})
    assert validated == {"q": None}


def test_validate_task_inputs_from_spec_rejects_default_none_on_plain_str():
    spec = make_task_spec(
        inputs=[input_field("q", desc="Question with null default.", default=None)],
        outputs=[output_field("a", desc="The answer.")],
        instructions="Answer.",
    )
    with pytest.raises(ValueError, match="got incompatible value None"):
        validate_task_inputs_from_spec(spec, {})
    with pytest.raises(ValueError, match="got incompatible value None"):
        validate_task_inputs_from_spec(spec, {"q": None})


def test_validate_task_inputs_from_spec_accepts_default_none_on_optional_str():
    spec = make_task_spec(
        inputs=[input_field("q", str | None, desc="Optional question.", default=None)],
        outputs=[output_field("a", desc="The answer.")],
        instructions="Answer.",
    )
    assert validate_task_inputs_from_spec(spec, {}) == {"q": None}
    assert validate_task_inputs_from_spec(spec, {"q": None}) == {"q": None}


def test_validate_task_inputs_from_spec_accepts_list_str():
    spec = make_task_spec(
        inputs=[input_field("tags", list[str], desc="Tag list.")],
        outputs=[output_field("a", desc="The answer.")],
        instructions="Answer.",
    )
    validated = validate_task_inputs_from_spec(spec, {"tags": ["a", "b"]})
    assert validated == {"tags": ["a", "b"]}


def test_validate_task_inputs_from_spec_rejects_invalid_list_str():
    spec = make_task_spec(
        inputs=[input_field("tags", list[str], desc="Tag list.")],
        outputs=[output_field("a", desc="The answer.")],
        instructions="Answer.",
    )
    with pytest.raises(ValueError, match="Type mismatch"):
        validate_task_inputs_from_spec(spec, {"tags": "not-a-list"})


def test_validate_task_inputs_from_spec_accepts_union_types():
    spec = make_task_spec(
        inputs=[input_field("value", str | int, desc="String or integer.")],
        outputs=[output_field("a", desc="The answer.")],
        instructions="Answer.",
    )
    assert validate_task_inputs_from_spec(spec, {"value": "text"}) == {"value": "text"}
    assert validate_task_inputs_from_spec(spec, {"value": 42}) == {"value": 42}


def test_validate_task_inputs_from_spec_accepts_literal():
    spec = make_task_spec(
        inputs=[input_field("level", Literal["low", "high"], desc="Level.")],
        outputs=[output_field("a", desc="The answer.")],
        instructions="Answer.",
    )
    assert validate_task_inputs_from_spec(spec, {"level": "low"}) == {"level": "low"}


def test_validate_task_inputs_from_spec_rejects_invalid_literal():
    spec = make_task_spec(
        inputs=[input_field("level", Literal["low", "high"], desc="Level.")],
        outputs=[output_field("a", desc="The answer.")],
        instructions="Answer.",
    )
    with pytest.raises(ValueError, match="Type mismatch"):
        validate_task_inputs_from_spec(spec, {"level": "medium"})


def test_validate_task_inputs_from_spec_accepts_custom_image_type():
    image = Image("https://example.com/test.png")
    spec = make_task_spec(
        inputs=[input_field("image", Image, desc="Input image.")],
        outputs=[output_field("a", desc="The answer.")],
        instructions="Answer.",
    )
    assert validate_task_inputs_from_spec(spec, {"image": image}) == {"image": image}


def test_validate_task_inputs_from_spec_rejects_invalid_image_type():
    spec = make_task_spec(
        inputs=[input_field("image", Image, desc="Input image.")],
        outputs=[output_field("a", desc="The answer.")],
        instructions="Answer.",
    )
    with pytest.raises(ValueError, match="Type mismatch"):
        validate_task_inputs_from_spec(spec, {"image": "not-an-image"})
