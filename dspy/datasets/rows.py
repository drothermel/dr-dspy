from collections.abc import Iterable, Mapping, Sequence

from dspy.primitives.example import Example


def rows_to_examples(
    rows: Iterable[Mapping[str, object]], fields: Sequence[str] | None, input_keys: tuple[str, ...]
) -> list[Example]:
    rows_list = list(rows)
    if not rows_list:
        return []
    resolved_fields = list(fields) if fields is not None else list(rows_list[0])
    return [
        Example.from_record({field: row[field] for field in resolved_fields}, input_keys=input_keys)
        for row in rows_list
    ]
