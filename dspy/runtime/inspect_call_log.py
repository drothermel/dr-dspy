"""Debug-only call log pretty-printing. Not part of the runtime spine."""

from __future__ import annotations

import json
import sys
from contextlib import suppress
from typing import TYPE_CHECKING, Any, TextIO, cast

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


def pretty_print_call_log(call_log: Sequence[CallRecord], n: int = 1, file: TextIO | None = None) -> None:
    out = file or sys.stdout
    use_colors = file is None

    def print_tool_calls(tool_calls: list[dict[str, Any]] | None) -> None:
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

    for item in call_log[-n:]:
        messages = item.messages_as_openai
        if not messages and item.prompt is not None:
            messages = [{"role": "user", "content": item.prompt}]
        outputs = item.outputs
        timestamp = item.timestamp
        print("\n\n\n", file=out)
        print(_blue(f"[{timestamp}]", use_colors=use_colors), file=out)
        for msg in messages:
            print(_red(f"{str(msg['role']).capitalize()} message:", use_colors=use_colors), file=out)
            if isinstance(msg["content"], str):
                print(msg["content"].strip(), file=out)
            elif isinstance(msg["content"], list):
                for c in msg["content"]:
                    if c["type"] == "text":
                        print(c["text"].strip(), file=out)
                    elif c["type"] == "image_url":
                        image_str = ""
                        url = c["image_url"].get("url", "")
                        if "base64," in url:
                            prefix, payload = url.split("base64,", 1)
                            len_base64 = len(payload)
                            image_str = f"<{prefix}base64,<IMAGE BASE 64 ENCODED({len_base64!s})>"
                        else:
                            image_str = f"<image_url: {url}>"
                        print(_blue(image_str.strip(), use_colors=use_colors), file=out)
                    elif c["type"] == "input_audio":
                        audio_format = c["input_audio"]["format"]
                        len_audio = len(c["input_audio"]["data"])
                        audio_str = f"<audio format='{audio_format}' base64-encoded, length={len_audio}>"
                        print(_blue(audio_str.strip(), use_colors=use_colors), file=out)
                    elif c["type"] == "file" or c["type"] == "input_file":
                        file_info = c.get("file", c.get("input_file", {}))
                        filename = file_info.get("filename", "")
                        file_id = file_info.get("file_id", "")
                        file_data = file_info.get("file_data", "")
                        file_str = f"<file: name:{filename}, id:{file_id}, data_length:{len(file_data)}>"
                        print(_blue(file_str.strip(), use_colors=use_colors), file=out)
            print_tool_calls(cast("list[dict[str, Any]] | None", msg.get("tool_calls")))
            print("\n", file=out)
        if not outputs:
            continue
        if isinstance(outputs[0], dict):
            if outputs[0].get("text"):
                print(_red("Response:", use_colors=use_colors), file=out)
                print(_green(outputs[0]["text"].strip(), use_colors=use_colors), file=out)
            print_tool_calls(outputs[0].get("tool_calls"))
        else:
            print(_red("Response:", use_colors=use_colors), file=out)
            print(_green(outputs[0].strip(), use_colors=use_colors), file=out)
        if len(outputs) > 1:
            choices_text = f" \t (and {len(outputs) - 1} other completions)"
            print(_red(choices_text, end="", use_colors=use_colors), file=out)
    print("\n\n\n", file=out)
