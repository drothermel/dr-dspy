"""Outer optimizer study over the unified v1 eval pipeline.

Select the generation shape with ``STUDY_PIPELINE`` (``direct`` |
``enc-dec``) and the optimizer with ``STUDY_STRATEGY`` (``grid`` |
``copro``). ``grid`` evaluates a fixed instruction list in one round
(proves the loop); ``copro`` runs coordinate ascent over the optimized
node's instruction (encoder for enc-dec, solver for direct) using logged
proposers with ``prompt_model != task_model``.
"""

from __future__ import annotations

import os

from dr_dspy.experiment_spec import (
    FieldSpec,
    GraphSpec,
    NodeConfig,
    NodeSpec,
)
from dr_dspy.humaneval_eval_dbos import EvalHumanEvalExperimentConfig
from dr_dspy.humaneval_study_dbos import (
    COPRO_STRATEGY,
    GRID_STRATEGY,
    StudyPlanConfig,
    create_study_app,
)
from dr_dspy.lm_utils import ModelConfig
from dr_dspy.runtime import run_typer_app

DATASET_NAME = "evalplus/humanevalplus"
DATASET_SPLIT = "test"
DEFAULT_SEED = 0
DEFAULT_SUBPROCESS_TIMEOUT = 15.0

TASK_MODEL = ModelConfig(
    model="openai/gpt-5.4-nano", reasoning={"effort": "none"}
)
PROMPT_MODEL = ModelConfig(model="moonshotai/kimi-k2", reasoning={})

SOLVE_INSTRUCTION = (
    "Write functional code in Python according to the prompt. "
    "Output only the code solution."
)
ENCODER_INSTRUCTION = (
    "Encode this Python function implementation into a complete "
    "lossless description. Preserve all behavior needed to reconstruct "
    "the code, but do not output Python code."
)
DECODER_INSTRUCTION = (
    "Decode the description into functional Python code. "
    "Output only Python code."
)

SOLVE_GRID_INSTRUCTIONS = (
    SOLVE_INSTRUCTION,
    "You are an expert Python programmer. Read the prompt and return a "
    "correct, efficient implementation. Output only the code.",
)
ENCODER_GRID_INSTRUCTIONS = (
    ENCODER_INSTRUCTION,
    "Summarize this Python function as a concise but complete natural-"
    "language specification sufficient to reconstruct it exactly. Do not "
    "emit Python code.",
)


def direct_graph(model: ModelConfig) -> GraphSpec:
    solve = NodeSpec(
        id="solve",
        op="llm_call",
        config=NodeConfig(
            model=model.model,
            temperature=0.0,
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


def encdec_graph(model: ModelConfig) -> GraphSpec:
    encode = NodeSpec(
        id="encode",
        op="llm_call",
        config=NodeConfig(
            model=model.model,
            temperature=0.0,
            reasoning=dict(model.reasoning),
            instruction=ENCODER_INSTRUCTION,
            signature_name="EncodeCode",
            fields=(
                FieldSpec(name="code", role="input"),
                FieldSpec(name="description", role="output"),
            ),
            output_field="description",
            input_bindings={"code": "task.ground_truth_code"},
        ),
    )
    decode = NodeSpec(
        id="decode",
        op="llm_call",
        config=NodeConfig(
            model=model.model,
            temperature=0.0,
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


def _build_configs() -> tuple[EvalHumanEvalExperimentConfig, StudyPlanConfig]:
    pipeline = os.environ.get("STUDY_PIPELINE", "enc-dec").lower()
    strategy = os.environ.get("STUDY_STRATEGY", GRID_STRATEGY).lower()
    if strategy not in (GRID_STRATEGY, COPRO_STRATEGY):
        raise ValueError(
            f"STUDY_STRATEGY must be {GRID_STRATEGY} or {COPRO_STRATEGY}"
        )
    if pipeline == "direct":
        base_graph = direct_graph(TASK_MODEL)
        plan = StudyPlanConfig(
            node_id="solve",
            base_graph=base_graph,
            base_instruction=SOLVE_INSTRUCTION,
            default_strategy=strategy,
            grid_instructions=SOLVE_GRID_INSTRUCTIONS,
            prompt_model=PROMPT_MODEL,
            default_breadth=4,
            default_depth=3,
            default_repetitions=1,
            default_val=16,
            default_test=16,
        )
        script_kind = "humaneval_study_dbos_v1_direct"
        max_tokens = 1000
    elif pipeline == "enc-dec":
        base_graph = encdec_graph(TASK_MODEL)
        plan = StudyPlanConfig(
            node_id="encode",
            base_graph=base_graph,
            base_instruction=ENCODER_INSTRUCTION,
            default_strategy=strategy,
            grid_instructions=ENCODER_GRID_INSTRUCTIONS,
            prompt_model=PROMPT_MODEL,
            default_breadth=4,
            default_depth=3,
            default_repetitions=3,
            default_val=16,
            default_test=16,
        )
        script_kind = "humaneval_study_dbos_v1_encdec"
        max_tokens = 2000
    else:
        raise ValueError(
            f"STUDY_PIPELINE must be 'direct' or 'enc-dec'; got {pipeline!r}"
        )
    eval_config = EvalHumanEvalExperimentConfig(
        pipeline=pipeline,
        script_kind=script_kind,
        dataset_name=DATASET_NAME,
        dataset_split=DATASET_SPLIT,
        graphs=(base_graph,),
        default_sample_count=16,
        default_seed=DEFAULT_SEED,
        default_repetitions=plan.default_repetitions,
        default_max_completion_tokens=max_tokens,
        default_subprocess_timeout=DEFAULT_SUBPROCESS_TIMEOUT,
    )
    return eval_config, plan


_EVAL_CONFIG, _PLAN_CONFIG = _build_configs()
app = create_study_app(_EVAL_CONFIG, _PLAN_CONFIG)

if __name__ == "__main__":
    run_typer_app(app)
