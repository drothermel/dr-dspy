from __future__ import annotations

import re


def is_openai_reasoning_model(model: str | None) -> bool:
    if not isinstance(model, str):
        return False
    model_family = model.split("/")[-1].lower() if "/" in model else model.lower()
    # Intentionally conservative regex for OpenAI reasoning-model naming; update when
    # OpenAI adds families. Keep aligned with LM._get_initial_kwargs and
    # serialize._validate_openai_reasoning_temperature.
    return (
        re.match(
            r"^(?:o[1345](?:-(?:mini|nano|pro))?(?:-\d{4}-\d{2}-\d{2})?|gpt-5(?!-chat)(?:-.*)?)$",
            model_family,
        )
        is not None
    )
