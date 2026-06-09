"""Debug-only call log pretty-printing. Not part of the runtime spine."""

from __future__ import annotations

import json
import sys
from contextlib import suppress
from typing import TYPE_CHECKING, Any, TextIO, cast

from dspy.clients.openai_format.chat_request import request_messages_as_openai

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dspy.core.types import CallRecord


def _green(text: str, end: str = "\n", *, use_colors: bool = True) -> str:
    if use_colors:
        return "\x1b[32m" + str(text).lstrip() + "\x1b[0m" + end
    return str(text).lstrip() + end


def _red(text: str, end: str = "\n", *, use_colors: bool = True) -> str:
    if use_colors:
        return "\x1b[31m" + str(text) + "\x1b[0m" + end
    return str(text) + end


def _blue(text: str, end: str = "\n", *, use_colors: bool = True) -> str:
    if use_colors:
        return "\x1b[34m" + str(text) + "\x1b[0m" + end
    return str(text) + end


def _print_tool_calls(
    tool_calls: list[dict[str, Any]] | None,
    *,
    out: TextIO,
    use_colors: bool,
) -> None:
    if tool_calls:
        print(_red("Tool calls:", use_colors=use_colors), file=out)
    for tool_call in tool_calls or []:
        function = tool_call.get("function") or {}
        arguments = function.get("arguments")
        arguments = tool_call.get("args", tool_call.get("arguments", {})) if arguments is None else arguments
        with suppress(json.JSONDecodeError):
            arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
        print(
            _green(
                f"{function.get('name') or tool_call.get('name', '<unknown>')}: {(json.dumps(arguments, ensure_ascii=False) if isinstance(arguments, (dict, list)) else str(arguments))}",
                use_colors=use_colors,
            ),
            file=out,
        )


def _print_message_content(content: Any, *, out: TextIO, use_colors: bool) -> None:
    if content is None:
        return
    if isinstance(content, str):
        print(content.strip(), file=out)
        return
    if not isinstance(content, list):
        return
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type == "text":
            print(part.get("text", "").strip(), file=out)
        elif part_type == "image_url":
            image_url = part.get("image_url", {})
            url = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url)
            if "base64," in url:
                prefix, payload = url.split("base64,", 1)
                image_str = f"<{prefix}base64,<IMAGE BASE 64 ENCODED({len(payload)!s})>"
            elif part.get("redacted"):
                image_str = "<image_url: redacted>"
            else:
                image_str = f"<image_url: {url}>"
            print(_blue(image_str.strip(), use_colors=use_colors), file=out)
        elif part_type == "input_audio":
            input_audio = part.get("input_audio", {})
            audio_format = input_audio.get("format", "") if isinstance(input_audio, dict) else ""
            len_audio = len(input_audio.get("data", "")) if isinstance(input_audio, dict) else 0
            audio_str = f"<audio format='{audio_format}' base64-encoded, length={len_audio}>"
            print(_blue(audio_str.strip(), use_colors=use_colors), file=out)
        elif part_type in {"file", "input_file"}:
            file_info = part.get("file", part.get("input_file", {}))
            filename = file_info.get("filename", "") if isinstance(file_info, dict) else ""
            file_id = file_info.get("file_id", "") if isinstance(file_info, dict) else ""
            file_data = file_info.get("file_data", "") if isinstance(file_info, dict) else ""
            file_str = f"<file: name:{filename}, id:{file_id}, data_length:{len(file_data)}>"
            print(_blue(file_str.strip(), use_colors=use_colors), file=out)


def _print_openai_messages(messages: list[dict[str, Any]], *, out: TextIO, use_colors: bool) -> None:
    for msg in messages:
        role = str(msg.get("role", "unknown")).capitalize()
        print(_red(f"{role} message:", use_colors=use_colors), file=out)
        _print_message_content(msg.get("content"), out=out, use_colors=use_colors)
        _print_tool_calls(cast("list[dict[str, Any]] | None", msg.get("tool_calls")), out=out, use_colors=use_colors)
        print("\n", file=out)


def _print_outputs(outputs: list[Any], *, out: TextIO, use_colors: bool) -> None:
    if not outputs:
        return
    if isinstance(outputs[0], dict):
        if outputs[0].get("text"):
            print(_red("Response:", use_colors=use_colors), file=out)
            print(_green(str(outputs[0]["text"]).strip(), use_colors=use_colors), file=out)
        _print_tool_calls(outputs[0].get("tool_calls"), out=out, use_colors=use_colors)
    else:
        print(_red("Response:", use_colors=use_colors), file=out)
        print(_green(str(outputs[0]).strip(), use_colors=use_colors), file=out)
    if len(outputs) > 1:
        choices_text = f" \t (and {len(outputs) - 1} other completions)"
        print(_red(choices_text, end="", use_colors=use_colors), file=out)


def pretty_print_disk_call_log(records: Sequence[dict[str, Any]], n: int = 1, file: TextIO | None = None) -> None:
    out = file or sys.stdout
    use_colors = file is None
    for item in records[-n:]:
        print("\n\n\n", file=out)
        print(_blue(f"[{item.get('timestamp', '')}]", use_colors=use_colors), file=out)
        messages = item.get("messages")
        if isinstance(messages, list):
            _print_openai_messages(messages, out=out, use_colors=use_colors)
        response = item.get("response")
        outputs = response.get("outputs") if isinstance(response, dict) else None
        if isinstance(outputs, list):
            _print_outputs(outputs, out=out, use_colors=use_colors)
    print("\n\n\n", file=out)


def pretty_print_call_log(call_log: Sequence[CallRecord], n: int = 1, file: TextIO | None = None) -> None:
    out = file or sys.stdout
    use_colors = file is None

    for item in call_log[-n:]:
        messages = request_messages_as_openai(item.request)
        if not messages and item.prompt is not None:
            messages = [{"role": "user", "content": item.prompt}]
        outputs = item.outputs
        timestamp = item.timestamp
        print("\n\n\n", file=out)
        print(_blue(f"[{timestamp}]", use_colors=use_colors), file=out)
        _print_openai_messages(messages, out=out, use_colors=use_colors)
        _print_outputs(list(outputs), out=out, use_colors=use_colors)
    print("\n\n\n", file=out)
