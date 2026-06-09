import pytest

from dspy.adapters.types.audio import _normalize_audio_format


@pytest.mark.parametrize(
    ("input_format", "expected_format"),
    [
        ("wav", "wav"),
        ("mp3", "mp3"),
        ("x-wav", "wav"),
        ("x-mp3", "mp3"),
        ("x-flac", "flac"),
        ("my-x-format", "my-x-format"),
        ("x-my-format", "my-format"),
        ("", ""),
        ("x-", ""),
    ],
)
def test_normalize_audio_format(input_format, expected_format):
    assert _normalize_audio_format(input_format) == expected_format
