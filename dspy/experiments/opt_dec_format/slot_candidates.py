"""Bounded decoder slot candidate models."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictBaseModel(BaseModel):
    """Strict base model for experiment-owned payloads."""

    model_config = ConfigDict(extra="forbid")


class SlotCapPolicy(StrictBaseModel):
    """Render-time slot character cap policy."""

    max_chars: int = Field(default=100, gt=0)
    over_limit_policy: Literal[
        "truncate_for_rendering_and_record_raw",
        "fail_config_validation",
    ] = "truncate_for_rendering_and_record_raw"


class SlotBundle(StrictBaseModel):
    """Raw bounded decoder slot values proposed by an optimizer."""

    task_instructions: str
    output_instructions: str
    failure_avoidance: str

    @field_validator(
        "task_instructions",
        "output_instructions",
        "failure_avoidance",
    )
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            msg = "Slot values must not be empty."
            raise ValueError(msg)
        return value

    def render(self, policy: SlotCapPolicy | None = None) -> RenderedSlotBundle:
        """Render slots under the cap policy while preserving raw values."""
        active = policy or SlotCapPolicy()
        rendered: dict[str, str] = {}
        truncated: dict[str, bool] = {}
        for name, value in self.model_dump().items():
            is_over = len(value) > active.max_chars
            if is_over and active.over_limit_policy == "fail_config_validation":
                msg = f"Slot {name!r} exceeds {active.max_chars} characters."
                raise ValueError(msg)
            rendered[name] = value[: active.max_chars] if is_over else value
            truncated[name] = is_over
        return RenderedSlotBundle(
            raw=self,
            rendered=SlotBundle.model_validate(rendered),
            truncated=truncated,
            policy=active,
        )

    def deterministic_id(
        self,
        *,
        optimizer_run_id: str,
        round_index: int,
        proposal_index: int,
        candidate_surface: str,
        template_id: str,
    ) -> str:
        """Derive a stable candidate id from slot content and context."""
        payload = {
            "candidate_surface": candidate_surface,
            "optimizer_run_id": optimizer_run_id,
            "proposal_index": proposal_index,
            "raw_slot_values": self.model_dump(),
            "round_index": round_index,
            "template_id": template_id,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
        return f"{optimizer_run_id}.r{round_index}.p{proposal_index}.{digest}"


class RenderedSlotBundle(StrictBaseModel):
    """Raw and rendered slot values plus cap metadata."""

    raw: SlotBundle
    rendered: SlotBundle
    truncated: dict[str, bool]
    policy: SlotCapPolicy
