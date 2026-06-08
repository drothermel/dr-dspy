import base64
import io
import mimetypes
from pathlib import Path
from typing import Any, cast

import pydantic

from dspy.adapters.types.base_type import Type

try:
    import soundfile as sf  # ty: ignore[unresolved-import]

    SF_AVAILABLE = True
except ImportError:
    SF_AVAILABLE = False


def _normalize_audio_format(audio_format: str) -> str:
    """Removes 'x-' prefixes from audio format strings."""
    return audio_format.removeprefix("x-")


class Audio(Type):
    data: str
    audio_format: str

    model_config = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
    )

    def format(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "input_audio",
                "input_audio": {
                    "data": self.data,
                    "format": self.audio_format,
                },
            }
        ]

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_input(cls, values: object) -> object:
        """
        Validate input for Audio, expecting 'data' and 'audio_format' keys in dictionary.
        """
        if isinstance(values, cls):
            return {"data": values.data, "audio_format": values.audio_format}
        return encode_audio(values)

    @classmethod
    def from_url(cls, url: str) -> "Audio":
        """
        Download an audio file from URL and encode it as base64.
        """
        import requests

        response = requests.get(url, timeout=30)
        response.raise_for_status()
        mime_type = response.headers.get("Content-Type", "audio/wav")
        if not mime_type.startswith("audio/"):
            raise ValueError(f"Unsupported MIME type for audio: {mime_type}")
        audio_format = mime_type.split("/")[1]

        audio_format = _normalize_audio_format(audio_format)

        encoded_data = base64.b64encode(response.content).decode("utf-8")
        return cls(data=encoded_data, audio_format=audio_format)

    @classmethod
    def from_file(cls, file_path: str) -> "Audio":
        """
        Read local audio file and encode it as base64.
        """
        path = Path(file_path)
        if not path.is_file():
            raise ValueError(f"File not found: {file_path}")

        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type or not mime_type.startswith("audio/"):
            raise ValueError(f"Unsupported MIME type for audio: {mime_type}")

        with path.open("rb") as file:
            file_data = file.read()

        audio_format = mime_type.split("/")[1]

        audio_format = _normalize_audio_format(audio_format)

        encoded_data = base64.b64encode(file_data).decode("utf-8")
        return cls(data=encoded_data, audio_format=audio_format)

    @classmethod
    def from_array(cls, array: object, sampling_rate: int, format: str = "wav") -> "Audio":
        """
        Process numpy-like array and encode it as base64. Uses sampling rate and audio format for encoding.
        """
        if not SF_AVAILABLE:
            raise ImportError("soundfile is required to process audio arrays.")

        byte_buffer = io.BytesIO()
        sf.write(
            byte_buffer,
            array,
            sampling_rate,
            format=format.upper(),
            subtype="PCM_16",
        )
        encoded_data = base64.b64encode(byte_buffer.getvalue()).decode("utf-8")
        return cls(data=encoded_data, audio_format=format)

    def __str__(self) -> str:
        return str(self.serialize_model())

    def __repr__(self) -> str:
        length = len(self.data)
        return f"Audio(data=<AUDIO_BASE_64_ENCODED({length})>, audio_format='{self.audio_format}')"


def encode_audio(audio: object, sampling_rate: int = 16000, format: str = "wav") -> dict[str, str]:
    """
    Encode audio to a dict with 'data' and 'audio_format'.

    Accepts: local file path, URL, data URI, dict, Audio instance, numpy array, or bytes (with known format).
    """
    if isinstance(audio, dict) and "data" in audio and "audio_format" in audio:
        return cast("dict[str, str]", audio)
    if isinstance(audio, Audio):
        return {"data": audio.data, "audio_format": audio.audio_format}
    if isinstance(audio, str) and audio.startswith("data:audio/"):
        try:
            header, b64data = audio.split(",", 1)
            mime = header.split(";")[0].split(":")[1]
            audio_format = mime.split("/")[1]

            audio_format = _normalize_audio_format(audio_format)

        except (IndexError, ValueError) as e:
            raise ValueError(f"Malformed audio data URI: {e}") from e
        return {"data": b64data, "audio_format": audio_format}
    if isinstance(audio, str) and Path(audio).is_file():
        a = Audio.from_file(audio)
        return {"data": a.data, "audio_format": a.audio_format}
    if isinstance(audio, str) and audio.startswith("http"):
        a = Audio.from_url(audio)
        return {"data": a.data, "audio_format": a.audio_format}
    if SF_AVAILABLE and hasattr(audio, "shape"):
        a = Audio.from_array(audio, sampling_rate=sampling_rate, format=format)
        return {"data": a.data, "audio_format": a.audio_format}
    if isinstance(audio, bytes):
        encoded = base64.b64encode(audio).decode("utf-8")
        return {"data": encoded, "audio_format": format}
    raise ValueError(f"Unsupported type for encode_audio: {type(audio)}")
