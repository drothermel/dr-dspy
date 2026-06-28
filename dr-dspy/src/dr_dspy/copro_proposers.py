"""COPRO instruction proposers, reimplemented as logged ops.

Stock COPRO (``dspy/teleprompt/copro_optimizer.py``) proposes instruction
text via a prompt model and throws the calls away. We reimplement its two
proposal signatures as our own ops so every proposer call is a logged
OpenRouter inference (usage + cost captured into candidate provenance),
keeping COPRO's ``prompt_model != task_model`` separation. Instruction
only — we intentionally drop COPRO's output-field-prefix tuning.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

import dspy
from dr_dspy import dspy_runner
from dr_dspy.lm_utils import LmEventBuffer, ModelConfig

DEFAULT_PROPOSAL_TEMPERATURE = 1.0
DEFAULT_PROPOSAL_MAX_TOKENS = 1000


class BasicGenerateInstruction(dspy.Signature):
    """Propose a new, improved instruction for a language-model program.

    You are given an initial instruction. Write a single better
    instruction that is clear, specific, and likely to improve the
    program's outputs. Output only the instruction text."""

    basic_instruction = dspy.InputField(
        desc="The initial instruction before optimization."
    )
    proposed_instruction = dspy.OutputField(
        desc="The improved instruction text."
    )


class GenerateInstructionGivenAttempts(dspy.Signature):
    """Propose a new instruction given previously attempted instructions
    and their scores.

    You are given a list of prior instructions and the score each
    achieved. Propose a single new instruction that is likely to score
    higher than all of them. Output only the instruction text."""

    attempted_instructions = dspy.InputField(
        desc="Prior instructions and their scores, best first."
    )
    proposed_instruction = dspy.OutputField(
        desc="A new instruction expected to outperform the attempts."
    )


class ProposedInstruction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instruction: str
    usage: dict = {}
    cost: float | None = None
    raw: str = ""


def format_attempts(history: Sequence[tuple[str, float]]) -> str:
    """COPRO-style rendering of (instruction, score) attempts."""
    lines: list[str] = []
    for index, (instruction, score) in enumerate(history, start=1):
        lines.append(f"Instruction #{index}: {instruction}")
        lines.append(f"Score #{index}: {score:.4f}")
    return "\n".join(lines)


def _run_proposal(
    *,
    signature: type[dspy.Signature],
    input_kwargs: dict,
    prompt_model: ModelConfig,
    temperature: float,
    max_completion_tokens: int,
    client: object | None,
) -> ProposedInstruction:
    buffer = LmEventBuffer()
    lm = dspy_runner.build_logged_lm(
        model=prompt_model.model,
        reasoning=prompt_model.reasoning,
        temperature=temperature,
        event_buffer=buffer,
        max_completion_tokens=max_completion_tokens,
        client=client,
    )
    text = dspy_runner.run_predictor(
        signature=signature,
        input_kwargs=input_kwargs,
        output_field="proposed_instruction",
        lm=lm,
        event_buffer=buffer,
    )
    result = dspy_runner.predictor_run_result(text, buffer)
    return ProposedInstruction(
        instruction=result.text.strip(),
        usage=result.usage_metadata,
        cost=result.provider_cost,
        raw=result.text,
    )


def propose_basic(
    *,
    prompt_model: ModelConfig,
    basic_instruction: str,
    breadth: int,
    temperature: float = DEFAULT_PROPOSAL_TEMPERATURE,
    max_completion_tokens: int = DEFAULT_PROPOSAL_MAX_TOKENS,
    client: object | None = None,
) -> list[ProposedInstruction]:
    """``breadth`` independent logged proposals seeded from one
    instruction (round 0 of coordinate ascent)."""
    return [
        _run_proposal(
            signature=BasicGenerateInstruction,
            input_kwargs={"basic_instruction": basic_instruction},
            prompt_model=prompt_model,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            client=client,
        )
        for _ in range(breadth)
    ]


def propose_given_attempts(
    *,
    prompt_model: ModelConfig,
    history: Sequence[tuple[str, float]],
    breadth: int,
    temperature: float = DEFAULT_PROPOSAL_TEMPERATURE,
    max_completion_tokens: int = DEFAULT_PROPOSAL_MAX_TOKENS,
    client: object | None = None,
) -> list[ProposedInstruction]:
    """``breadth`` logged proposals conditioned on the sorted attempt
    history (later coordinate-ascent rounds)."""
    attempts = format_attempts(history)
    return [
        _run_proposal(
            signature=GenerateInstructionGivenAttempts,
            input_kwargs={"attempted_instructions": attempts},
            prompt_model=prompt_model,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            client=client,
        )
        for _ in range(breadth)
    ]
