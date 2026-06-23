from __future__ import annotations

import pytest

from dspy.experiments.opt_dec_format.slot_candidates import SlotBundle, SlotCapPolicy


def test_slot_bundle_preserves_raw_and_truncates_rendered() -> None:
    bundle = SlotBundle(
        task_instructions="Implement carefully",
        output_instructions="Only code",
        failure_avoidance="x" * 12,
    )

    rendered = bundle.render(SlotCapPolicy(max_chars=5))

    assert rendered.raw.failure_avoidance == "x" * 12
    assert rendered.rendered.failure_avoidance == "x" * 5
    assert rendered.truncated["failure_avoidance"] is True


def test_slot_bundle_rejects_empty_values() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        SlotBundle(
            task_instructions=" ",
            output_instructions="Only code",
            failure_avoidance="Avoid stubs",
        )


def test_candidate_id_is_deterministic() -> None:
    bundle = SlotBundle(
        task_instructions="Implement carefully",
        output_instructions="Only code",
        failure_avoidance="Avoid stubs",
    )

    first = bundle.deterministic_id(
        optimizer_run_id="run",
        round_index=0,
        proposal_index=1,
        candidate_surface="bounded_slots",
        template_id="template",
    )
    second = bundle.deterministic_id(
        optimizer_run_id="run",
        round_index=0,
        proposal_index=1,
        candidate_surface="bounded_slots",
        template_id="template",
    )

    assert first == second
