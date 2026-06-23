"""Prompt rendering helpers for decoder-format experiments."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from dspy.experiments.opt_dec_format.slot_candidates import RenderedSlotBundle


class DecoderPromptInputs(BaseModel):
    """Inputs for description-only decoder prompt rendering."""

    model_config = ConfigDict(extra="forbid")

    encoded_description: str


def render_template(
    template_text: str,
    *,
    inputs: DecoderPromptInputs,
    slots: RenderedSlotBundle | None = None,
) -> str:
    """Render a decoder template from explicit inputs and optional slots."""
    values: dict[str, str] = inputs.model_dump()
    if slots is not None:
        values.update(slots.rendered.model_dump())
    return template_text.format_map(values)


def render_template_file(
    path: Path | str,
    *,
    inputs: DecoderPromptInputs,
    slots: RenderedSlotBundle | None = None,
) -> str:
    """Read and render a Markdown decoder prompt template."""
    return render_template(
        Path(path).read_text(encoding="utf-8"),
        inputs=inputs,
        slots=slots,
    )
