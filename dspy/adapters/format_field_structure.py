from __future__ import annotations

FIELD_STRUCTURE_INTRO = (
    "All interactions will be structured in the following way, with the appropriate values filled in."
)


def build_field_structure_instructions(
    *,
    input_section: str,
    output_section: str,
    input_preamble: str | None = None,
    output_preamble: str | None = None,
    completed_marker: str | None = None,
) -> str:
    parts = [FIELD_STRUCTURE_INTRO]
    if input_preamble:
        parts.append(input_preamble)
    parts.append(input_section)
    if output_preamble:
        parts.append(output_preamble)
    parts.append(output_section)
    if completed_marker:
        parts.append(completed_marker)
    return "\n\n".join(parts).strip()
