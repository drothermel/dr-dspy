from typing import Literal

import pytest

from dspy.task_spec.type_format import format_type_annotation


@pytest.mark.parametrize(
    ("annotation", "diagnostic", "prompt"),
    [
        (int, "int", "int"),
        (list[str], "list[str]", "list[str]"),
        (str | None, "UnionType[str, NoneType]", "UnionType[str, NoneType]"),
        (tuple[int, ...], "tuple[int, ...]", "tuple[int, ...]"),
        (Literal["a", "b"], "Literal['a', 'b']", "Literal['a', 'b']"),
        (Literal["it's", 'say "hi"'], 'Literal["it\'s", \'say "hi"\']', 'Literal["it\'s", \'say "hi"\']'),
    ],
)
def test_format_type_annotation_modes(annotation, diagnostic, prompt):
    assert format_type_annotation(annotation) == diagnostic
    assert format_type_annotation(annotation, quote_string_literals=True) == prompt
