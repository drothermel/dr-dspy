Before commiting run:
```
uv run ruff check --fix
uv run ty check --fix
uv run ruff format
```

Then fix any remaining issues and reformat before committing.

## Async-only public API

DSPy modules, LMs, adapters, `Evaluate`, `Parallel`, and teleprompter `compile` are async.
Use `await` at call sites; in scripts use `asyncio.run(...)`.

```python
# Module invocation
result = await program(question="What is DSPy?")

# Evaluation
evaluator = Evaluate(devset=devset, metric=my_metric)
score = await evaluator(program)

# Parallel batch
parallel = Parallel(max_concurrency=8)
results = await parallel([(module, example), ...])

# Optimizers
compiled = await teleprompter.compile(student, trainset=trainset)
```

`Module.acall` and `BaseLM.acall` are compatibility aliases for `__call__`.

## Internal call-site conventions

- Use keyword arguments for multi-arg calls to DSPy-internal functions when meaning is not obvious from position.
- Do not add keyword-only `*` to public constructors or documented callback protocols (e.g. `metric(example, prediction, trace)`).
- Spine APIs require keywords at call sites: `run_bounded(items=..., fn=...)`, `adapter.acall(lm=..., config=..., signature=..., demos=..., inputs=...)`.
