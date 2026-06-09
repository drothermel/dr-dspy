# Persistence migration guide

Persistence lives under `dspy.persistence` with focused submodules:

- `metadata.py` — dependency versions, drift warnings, metadata envelope
- `state.py` — module state dump/apply and `.json`/`.pkl` file I/O
- `program.py` — whole-program cloudpickle directory I/O
- `embeddings.py` — retriever artifact directory I/O

`Module.save`, `Module.load`, `Module.dump_state`, and `Module.load_state` remain the primary user-facing entrypoints but delegate to these helpers.

## Breaking changes

| Old | New |
| --- | --- |
| `dspy.persistence.load(path, allow_pickle=True)` | `dspy.persistence.load_program(path, allow_pickle=True) -> Module` |
| `Embeddings.from_saved(path, embedder)` | `dspy.persistence.load_embeddings(path, embedder=embedder)` |
| `EmbeddingsWithScores.from_saved(...)` | `dspy.persistence.load_embeddings(..., retriever_cls=EmbeddingsWithScores)` |
| `Module.load(path)` returns `None` | returns `Self` |
| `Module.load` / `load_state` without `custom_types` | both accept optional `custom_types` |

## Public spine exports

```python
from dspy.persistence import (
    apply_module_state,
    dump_module_state,
    get_dependency_versions,
    load_embeddings,
    load_embeddings_into,
    load_program,
    load_state,
    save_embeddings,
    save_program,
    save_state,
)
```

## File formats (unchanged)

- **State JSON/PKL:** predictor state keys plus embedded `metadata.dependency_versions`
- **Program directory:** `program.pkl` + `metadata.json`
- **Embeddings directory:** `config.json`, `corpus_embeddings.npy`, optional `faiss_index.bin`

Pickle loads still require explicit `allow_pickle=True` on program and state `.pkl` paths.
