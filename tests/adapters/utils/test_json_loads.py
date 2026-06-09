import json

import pytest

from dspy.adapters.utils.json_loads import load_json


def test_load_json_strict_raises_on_malformed():
    with pytest.raises(json.JSONDecodeError):
        load_json("{not json", repair=False)


def test_load_json_repair_fixes_malformed_when_opted_in():
    result = load_json("{key: 'value'}", repair=True)
    assert result == {"key": "value"}
