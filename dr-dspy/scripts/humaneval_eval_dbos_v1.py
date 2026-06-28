"""Unified HumanEval eval experiment (v1, node-graph pipeline).

One script, both generation shapes. Select the pipeline with the
``EVAL_V1_PIPELINE`` env var: ``direct`` (default) or ``enc-dec``. The
sweep axes (models, budget ratios) are pre-enumerated into a list of
concrete ``GraphSpec`` objects; everything else flows through the shared
unified pipeline in ``dr_dspy.humaneval_eval_dbos``.
"""

from __future__ import annotations

import os

from dr_dspy.experiment_spec import (
    FieldSpec,
    GraphSpec,
    NodeConfig,
    NodeSpec,
)
from dr_dspy.humaneval_eval_dbos import (
    EvalHumanEvalExperimentConfig,
    create_app,
)
from dr_dspy.lm_utils import ModelConfig
from dr_dspy.runtime import run_typer_app

DATASET_NAME = "evalplus/humanevalplus"
DATASET_SPLIT = "test"
DEFAULT_SEED = 0
DEFAULT_SUBPROCESS_TIMEOUT = 15.0

SOLVE_INSTRUCTION = (
    "Write functional code in Python according to the prompt. "
    "Output only the code solution."
)
ENCODER_INSTRUCTION = (
    "Encode this Python function implementation into a complete "
    "lossless description. Preserve all behavior needed to reconstruct "
    "the code, but do not output Python code."
)
ENCODER_BUDGETED_INSTRUCTION = (
    ENCODER_INSTRUCTION
    + " Keep the description within at most max_characters characters."
)
DECODER_INSTRUCTION = (
    "Decode the description into functional Python code. "
    "Output only Python code."
)

DEFAULT_MODEL_CONFIGS: tuple[ModelConfig, ...] = (
    ModelConfig(model="openai/gpt-5.1-codex-mini", reasoning={}),
    ModelConfig(model="moonshotai/kimi-k2-0905", reasoning={}),
    ModelConfig(model="qwen/qwen3-coder-next", reasoning={}),
    ModelConfig(
        model="deepseek/deepseek-v3.1-terminus",
        reasoning={"enabled": False},
    ),
    ModelConfig(model="moonshotai/kimi-k2", reasoning={}),
    ModelConfig(model="z-ai/glm-4.7", reasoning={"enabled": False}),
    ModelConfig(model="z-ai/glm-5", reasoning={"enabled": False}),
    ModelConfig(
        model="deepseek/deepseek-v4-pro",
        reasoning={"enabled": False},
    ),
    ModelConfig(
        model="deepseek/deepseek-v4-flash",
        reasoning={"enabled": False},
    ),
    ModelConfig(model="mistralai/mistral-large-2512", reasoning={}),
    ModelConfig(model="openai/gpt-oss-120b", reasoning={"effort": "low"}),
    ModelConfig(model="mistralai/codestral-2508", reasoning={}),
    ModelConfig(model="qwen/qwen3-coder-flash", reasoning={}),
    ModelConfig(model="openai/gpt-5-nano", reasoning={"effort": "low"}),
    ModelConfig(
        model="deepseek/deepseek-chat-v3.1",
        reasoning={"enabled": False},
    ),
    ModelConfig(model="openai/gpt-5.4-nano", reasoning={"effort": "none"}),
)

DIRECT_TEMPERATURES: tuple[float, ...] = (0.0,)
ENCODER_TEMPERATURE = 0.0
DECODER_TEMPERATURE = 0.0
BUDGET_RATIOS: tuple[float | None, ...] = (
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    None,
)


def direct_graph(model: ModelConfig, temperature: float) -> GraphSpec:
    solve = NodeSpec(
        id="solve",
        op="llm_call",
        config=NodeConfig(
            model=model.model,
            temperature=temperature,
            reasoning=dict(model.reasoning),
            instruction=SOLVE_INSTRUCTION,
            signature_name="Solve",
            fields=(
                FieldSpec(name="prompt", role="input"),
                FieldSpec(name="code", role="output", type_name="code"),
            ),
            output_field="code",
            input_bindings={"prompt": "task.prompt"},
        ),
    )
    return GraphSpec(
        nodes=(solve,),
        terminal_node_id="solve",
        compression_source="task.prompt",
    )


def encdec_graph(
    model: ModelConfig,
    *,
    encoder_temperature: float,
    decoder_temperature: float,
    budget_ratio: float | None,
) -> GraphSpec:
    budgeted = budget_ratio is not None
    encoder_fields = [FieldSpec(name="code", role="input")]
    if budgeted:
        encoder_fields.append(
            FieldSpec(name="max_characters", role="input", type_name="int")
        )
    encoder_fields.append(FieldSpec(name="description", role="output"))
    encode = NodeSpec(
        id="encode",
        op="llm_call",
        config=NodeConfig(
            model=model.model,
            temperature=encoder_temperature,
            reasoning=dict(model.reasoning),
            instruction=(
                ENCODER_BUDGETED_INSTRUCTION
                if budgeted
                else ENCODER_INSTRUCTION
            ),
            signature_name=(
                "EncodeCodeBudgeted" if budgeted else "EncodeCode"
            ),
            fields=tuple(encoder_fields),
            output_field="description",
            input_bindings={"code": "task.ground_truth_code"},
            extra=({"budget_ratio": budget_ratio} if budgeted else {}),
        ),
    )
    decode = NodeSpec(
        id="decode",
        op="llm_call",
        config=NodeConfig(
            model=model.model,
            temperature=decoder_temperature,
            reasoning=dict(model.reasoning),
            instruction=DECODER_INSTRUCTION,
            signature_name="DecodeCode",
            fields=(
                FieldSpec(name="description", role="input"),
                FieldSpec(name="code", role="output", type_name="code"),
            ),
            output_field="code",
            input_bindings={"description": "encode"},
        ),
    )
    return GraphSpec(
        nodes=(encode, decode),
        terminal_node_id="decode",
        compression_source="encode",
    )


def direct_graphs() -> tuple[GraphSpec, ...]:
    return tuple(
        direct_graph(model, temperature)
        for model in DEFAULT_MODEL_CONFIGS
        for temperature in DIRECT_TEMPERATURES
    )


def encdec_graphs() -> tuple[GraphSpec, ...]:
    return tuple(
        encdec_graph(
            model,
            encoder_temperature=ENCODER_TEMPERATURE,
            decoder_temperature=DECODER_TEMPERATURE,
            budget_ratio=budget_ratio,
        )
        for model in DEFAULT_MODEL_CONFIGS
        for budget_ratio in BUDGET_RATIOS
    )


DIRECT_CONFIG = EvalHumanEvalExperimentConfig(
    pipeline="direct",
    script_kind="humaneval_eval_dbos_v1_direct",
    dataset_name=DATASET_NAME,
    dataset_split=DATASET_SPLIT,
    graphs=direct_graphs(),
    default_sample_count=10,
    default_seed=DEFAULT_SEED,
    default_repetitions=1,
    default_max_completion_tokens=1000,
    default_subprocess_timeout=DEFAULT_SUBPROCESS_TIMEOUT,
)

ENCDEC_CONFIG = EvalHumanEvalExperimentConfig(
    pipeline="enc-dec",
    script_kind="humaneval_eval_dbos_v1_encdec",
    dataset_name=DATASET_NAME,
    dataset_split=DATASET_SPLIT,
    graphs=encdec_graphs(),
    default_sample_count=164,
    default_seed=DEFAULT_SEED,
    default_repetitions=3,
    default_max_completion_tokens=2000,
    default_subprocess_timeout=DEFAULT_SUBPROCESS_TIMEOUT,
)

_CONFIGS = {"direct": DIRECT_CONFIG, "enc-dec": ENCDEC_CONFIG}
PIPELINE = os.environ.get("EVAL_V1_PIPELINE", "direct").lower()
if PIPELINE not in _CONFIGS:
    raise ValueError(
        f"EVAL_V1_PIPELINE must be one of {sorted(_CONFIGS)}; got {PIPELINE!r}"
    )

app = create_app(_CONFIGS[PIPELINE])

if __name__ == "__main__":
    run_typer_app(app)
