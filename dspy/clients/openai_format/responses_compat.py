from typing import Any

import pydantic


def convert_chat_request_to_responses_request(request: dict[str, Any]) -> dict[str, Any]:
    request = dict(request)
    if "messages" in request:
        input_items = []
        for msg in request.pop("messages"):
            content_blocks = []
            c = msg.get("content")
            if isinstance(c, str):
                content_blocks.append({"type": "input_text", "text": c})
            elif isinstance(c, list):
                content_blocks.extend(_convert_content_item_to_responses_format(item) for item in c)
            input_items.append({"role": msg.get("role", "user"), "content": content_blocks})
        request["input"] = input_items
    if "reasoning_effort" in request:
        effort = request.pop("reasoning_effort")
        request["reasoning"] = {"effort": effort, "summary": "auto"}
    if "response_format" in request:
        response_format = request.pop("response_format")
        if isinstance(response_format, type) and issubclass(response_format, pydantic.BaseModel):
            response_format = {
                "name": response_format.__name__,
                "type": "json_schema",
                "schema": response_format.model_json_schema(),
            }
        text = request.pop("text", {})
        request["text"] = {**text, "format": response_format}
    return request


def _convert_content_item_to_responses_format(item: dict[str, Any]) -> dict[str, Any]:
    if item.get("type") == "image_url":
        image_url = item.get("image_url", {}).get("url", "")
        return {"type": "input_image", "image_url": image_url}
    if item.get("type") == "text":
        return {"type": "input_text", "text": item.get("text", "")}
    if item.get("type") == "file":
        file = item.get("file", {})
        return {
            "type": "input_file",
            "file_data": file.get("file_data"),
            "filename": file.get("filename"),
            "file_id": file.get("file_id"),
        }
    return item
