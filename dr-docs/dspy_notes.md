# DSPy Notes

This is a log for strange things I find while working with DSPy.

## 2026-06-08 - Slow clone root cause

The repository took multiple minutes to clone because the Git history contains large generated artifacts, even though the current checkout is not especially large.

Local measurements showed a `.git` directory of about 178 MiB, with a packed object database of about 169 MiB. The current working tree was only about 201 MiB total and `HEAD` contained 530 files, so the clone cost is mostly historical object transfer rather than checkout size.

The main historical offenders were:

- `test_before_pypi/`: a committed Python virtualenv, about 96.66 MiB packed and 330.32 MiB raw across 7,788 blobs. It was introduced by `b29b55d4` and later removed by `3cb51264`, but normal clones still fetch those old objects.
- `cache/`: committed runtime/joblib cache artifacts, about 16.20 MiB packed and 50.58 MiB raw across 9,732 blobs. This included many `cache/joblib/.../output.pkl` files, added around `24f25456` and later removed in `65ba23a4`.
- `docs/`: legitimate documentation/media history, about 38.97 MiB packed, including large GIF assets.

The practical workaround is a shallow clone, for example `git clone --depth 1 ...`. A real repository-level fix would require rewriting history with something like `git filter-repo` or BFG to purge the generated artifacts, which would be disruptive for existing forks and clones.
