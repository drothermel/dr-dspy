# Adapter tests

## Tiers

1. **Contract** (`test_format_contract.py`, `test_multimodal_contract.py`, `call/test_*.py`) — message roles, field markers, multimodal block presence, kitchen-sink structural coverage, pipeline stage mutations.
2. **Golden** (`test_golden_prompts.py` + `golden/`) — full prompt snapshots where prompt text is the behavior under review.
3. **Integration** — end-to-end `acall` with `tests.test_utils.DummyLM` or mocked litellm (adapter-specific files).

## When to use which tier

| Situation | Tier |
|-----------|------|
| Adapter-specific prompt wording (Chat `[[ ## field ## ]]` vs JSON wire vs XML tags) | Golden — one `GoldenPromptCase` per adapter |
| Broad multimodal / kitchen-sink coverage | Contract — fragment or structural assertions |
| Pipeline postprocess, parse fallback, native adaptation | `call/` unit tests |
| End-to-end parse and tool-call loops | Integration in adapter test files |

## Layout

- `scenarios/` — shared `FormatScenarioCase` builders (`qa.py`, `history.py`, `multimodal.py`, `pydantic_cases.py`, per-adapter `*_cases.py`).
- `golden/` — per-adapter `GoldenPromptCase` tuples with verbatim `messages` and optional `lm_kwargs`; assembled in `registry.py`.
- `assertions.py` — exact match, citation schema normalization, multimodal block helpers.
- `test_golden_prompts.py` — single parametrized runner over `ALL_GOLDEN_CASES`.
- `call/` — pipeline, postprocess, stages, parse fallback, response format routing, wrappers, hierarchy.
- `types/tool/` — `Tool`, `ToolCalls`, schema helpers.
- `utils/` — parse-value and formatting utilities.
- `conftest.py` — `format_messages_and_lm_kwargs`, `CapturingLM`, `adapter_format_as_openai`.

## Adding a new golden scenario

1. Add a scenario builder in `scenarios/` returning `FormatScenarioCase`.
2. Add a `GoldenPromptCase` with exact `messages` (and `lm_kwargs` if relevant) in the matching `golden/<adapter>.py`.
3. Register via `golden/registry.py` if using a new module.
4. Run `uv run pytest tests/adapters/test_golden_prompts.py -n0 -v -k <case-id>`.

Do not add `format_exact_messages_*` tests to adapter-specific files.
