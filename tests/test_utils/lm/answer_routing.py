"""Answer lookup helpers for DummyLM."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from dspy.core.types import LMMessage

from dspy.adapters.format.prompt_sections import FIELD_HEADER_PATTERN

DummyAnswer = dict[str, Any]
DummyAnswers = list[DummyAnswer] | dict[str, DummyAnswer]


def follow_example_output(messages: Sequence[LMMessage]) -> str | None:
    fields = defaultdict(int)
    for message in messages:
        content = getattr(message, "text", None)
        if content and (ma := FIELD_HEADER_PATTERN.match(content)):
            fields[content[ma.start() : ma.end()]] += 1
    if not fields:
        return None
    max_count = max(fields.values())
    output_fields = [field for field, count in fields.items() if count != max_count]
    final_input = (messages[-1].text or "").split("\n\n")[0]
    for input_message, output_message in zip(reversed(messages[:-1]), reversed(messages), strict=False):
        input_content = getattr(input_message, "text", "") or ""
        output_content = getattr(output_message, "text", "") or ""
        if any(field in output_content for field in output_fields) and final_input in input_content:
            return output_content
    return None


def resolve_dict_answer(
    answers: dict[str, DummyAnswer],
    messages: Sequence[LMMessage],
    *,
    format_fields,
) -> Any:
    last_message = messages[-1]
    last_content = getattr(last_message, "text", None)
    if last_content is None and isinstance(last_message, dict):
        last_content = last_message.get("content")
    last_content_str = last_content if isinstance(last_content, str) else ""
    return next(
        (format_fields(answer) for key, answer in answers.items() if key in last_content_str),
        "No more responses",
    )


def resolve_sequential_answer(
    answer_iter: Iterator[DummyAnswer],
    *,
    format_fields,
) -> Any:
    return format_fields(next(answer_iter, {"answer": "No more responses"}))


def resolve_answer(
    *,
    answers: DummyAnswers | Iterator[DummyAnswer],
    messages: Sequence[LMMessage],
    follow_examples: bool,
    format_fields,
) -> Any:
    if follow_examples:
        return follow_example_output(messages)
    if isinstance(answers, dict):
        dict_answers = cast("dict[str, DummyAnswer]", answers)
        return resolve_dict_answer(dict_answers, messages, format_fields=format_fields)
    answer_iter = cast("Iterator[DummyAnswer]", answers)
    return resolve_sequential_answer(answer_iter, format_fields=format_fields)
