import base64
import io
import mimetypes
from functools import lru_cache
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import pydantic
from typing_extensions import override

from dspy.adapters.types.base_type import Type

try:
    from PIL import Image as PILImage

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class Image(Type):
    url: str
    model_config = pydantic.ConfigDict(frozen=True, str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    def __init__(self, url: object = None, *, download: bool = False, verify: bool = True, **data: object) -> None:
        if url is not None and "url" not in data:
            data["url"] = url
        if "url" in data:
            data["url"] = encode_image(data["url"], download_images=download, verify=verify)
        super().__init__(**data)

    def format(self) -> list[dict[str, Any]] | str:
        try:
            return _format_image_content(self.url)
        except Exception as e:
            raise ValueError(f"Failed to format image for DSPy: {e}") from e

    @override
    def __str__(self) -> str:
        return str(self.serialize_model())

    @override
    def __repr__(self) -> str:
        if "base64" in self.url:
            len_base64 = len(self.url.split("base64,")[1])
            image_type = self.url.split(";")[0].split("/")[-1]
            return f"Image(url=data:image/{image_type};base64,<IMAGE_BASE_64_ENCODED({len_base64!s})>)"
        return f"Image(url='{self.url}')"


def is_url(string: str) -> bool:
    try:
        result = urlparse(string)
        return all([result.scheme in ("http", "https", "gs"), result.netloc])
    except ValueError:
        return False


def encode_image(image: object, download_images: bool = False, verify: bool = True) -> str:
    if isinstance(image, dict) and "url" in image:
        image = cast("dict[str, object]", image)
        url = image["url"]
        return url if isinstance(url, str) else str(url)
    if isinstance(image, str):
        if image.startswith("data:"):
            return image
        if Path(image).is_file():
            return _encode_image_from_file(image)
        if is_url(image):
            if download_images:
                return _encode_image_from_url(image, verify=verify)
            return image
        raise ValueError(
            f"Unrecognized file string: {image}; If this file type should be supported, please open an issue."
        )
    if PIL_AVAILABLE and isinstance(image, PILImage.Image):
        return _encode_pil_image(image)
    if isinstance(image, bytes):
        if not PIL_AVAILABLE:
            raise ImportError("Pillow is required to process image bytes.")
        img = PILImage.open(io.BytesIO(image))
        return _encode_pil_image(img)
    if isinstance(image, Image):
        return image.url
    raise ValueError(f"Unsupported image type: {type(image)}")


def _encode_image_from_file(file_path: str) -> str:
    with Path(file_path).open("rb") as file:
        file_data = file.read()
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        raise ValueError(f"Could not determine MIME type for file: {file_path}")
    encoded_data = base64.b64encode(file_data).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_data}"


def _encode_image_from_url(image_url: str, verify: bool = True) -> str:
    import requests

    response = requests.get(image_url, verify=verify, timeout=30)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    if content_type:
        mime_type = content_type
    else:
        mime_type, _ = mimetypes.guess_type(image_url)
        if mime_type is None:
            raise ValueError(f"Could not determine MIME type for URL: {image_url}")
    encoded_data = base64.b64encode(response.content).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_data}"


def _encode_pil_image(image: "PILImage.Image") -> str:
    buffered = io.BytesIO()
    file_format = image.format or "PNG"
    image.save(buffered, format=file_format)
    file_extension = file_format.lower()
    mime_type, _ = mimetypes.guess_type(f"file.{file_extension}")
    if mime_type is None:
        raise ValueError(f"Could not determine MIME type for image format: {file_format}")
    encoded_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_data}"


def _get_file_extension(path_or_url: str) -> str:
    extension = Path(urlparse(path_or_url).path).suffix.lstrip(".").lower()
    return extension or "png"


def is_image(obj: object) -> bool:
    if PIL_AVAILABLE and isinstance(obj, PILImage.Image):
        return True
    return bool(isinstance(obj, str) and (obj.startswith("data:") or Path(obj).is_file() or is_url(obj)))


@lru_cache(maxsize=32)
def _format_image_content(url: str) -> list[dict[str, Any]]:
    image_url = encode_image(url)
    return [{"type": "image_url", "image_url": {"url": image_url}}]
