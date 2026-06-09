from dspy.utils.serialize import to_jsonable


def test_to_jsonable_handles_circular_references():
    data: dict[str, object] = {}
    data["self"] = data
    result = to_jsonable(data)
    assert result == {"self": "<circular>"}


def test_to_jsonable_converts_tuples_to_lists():
    assert to_jsonable((1, 2)) == [1, 2]
