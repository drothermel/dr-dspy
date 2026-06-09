from __future__ import annotations

import pydantic

from dspy.adapters.types.document import Document
from dspy.adapters.types.image import Image
from dspy.task_spec import input_field, make_task_spec, output_field
from tests.adapters.scenarios.case import FormatScenarioCase


def single_image_case() -> FormatScenarioCase:
    task_spec = make_task_spec(
        {"image": input_field("image", type_=Image, desc="The image."), "text": output_field("text", desc="The text.")},
        instructions="Given the fields `image`, produce the fields `text`.",
    )
    return FormatScenarioCase(
        task_spec=task_spec,
        demos=(),
        inputs={"image": Image(url="https://example.com/image.jpg")},
    )


def image_with_demos_case() -> FormatScenarioCase:
    task_spec = make_task_spec(
        {"image": input_field("image", type_=Image, desc="The image."), "text": output_field("text", desc="The text.")},
        instructions="Given the fields `image`, produce the fields `text`.",
    )
    return FormatScenarioCase(
        task_spec=task_spec,
        demos=(
            {"image": Image(url="https://example.com/image1.jpg"), "text": "This is a test image"},
            {"image": Image(url="https://example.com/image2.jpg"), "text": "This is another test image"},
        ),
        inputs={"image": Image(url="https://example.com/image3.jpg")},
    )


def nested_images_case() -> FormatScenarioCase:
    class ImageWrapper(pydantic.BaseModel):
        images: list[Image]
        tag: list[str]

    task_spec = make_task_spec(
        {
            "image": input_field("image", type_=ImageWrapper, desc="The image."),
            "text": output_field("text", desc="The text."),
        },
        instructions="Given the fields `image`, produce the fields `text`.",
    )
    image_wrapper = ImageWrapper(
        images=[
            Image(url="https://example.com/image1.jpg"),
            Image(url="https://example.com/image2.jpg"),
            Image(url="https://example.com/image3.jpg"),
        ],
        tag=["test", "example"],
    )
    return FormatScenarioCase(task_spec=task_spec, demos=(), inputs={"image": image_wrapper})


def nested_images_with_demos_case() -> FormatScenarioCase:
    class ImageWrapper(pydantic.BaseModel):
        images: list[Image]
        tag: list[str]

    task_spec = make_task_spec(
        {
            "image": input_field("image", type_=ImageWrapper, desc="The image."),
            "text": output_field("text", desc="The text."),
        },
        instructions="Given the fields `image`, produce the fields `text`.",
    )
    image_wrapper = ImageWrapper(
        images=[
            Image(url="https://example.com/image1.jpg"),
            Image(url="https://example.com/image2.jpg"),
            Image(url="https://example.com/image3.jpg"),
        ],
        tag=["test", "example"],
    )
    image_wrapper_2 = ImageWrapper(images=[Image(url="https://example.com/image4.jpg")], tag=["test", "example"])
    return FormatScenarioCase(
        task_spec=task_spec,
        demos=({"image": image_wrapper, "text": "This is a test image"},),
        inputs={"image": image_wrapper_2},
    )


def nested_documents_case() -> FormatScenarioCase:
    class DocumentWrapper(pydantic.BaseModel):
        documents: list[Document]

    task_spec = make_task_spec(
        {
            "document": input_field("document", type_=DocumentWrapper, desc="The document."),
            "text": output_field("text", desc="The text."),
        },
        instructions="Given the fields `document`, produce the fields `text`.",
    )
    document_wrapper = DocumentWrapper(
        documents=[Document(data="Hello, world!"), Document(data="Hello, world 2!")],
    )
    return FormatScenarioCase(task_spec=task_spec, demos=(), inputs={"document": document_wrapper})
