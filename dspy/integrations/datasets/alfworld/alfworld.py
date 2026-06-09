from __future__ import annotations

import contextlib
import importlib
import queue
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, cast

from dspy.datasets.dataset import Dataset

if TYPE_CHECKING:
    from types import TracebackType


class MessageQueue(Protocol):
    def get(self) -> object: ...

    def put(self, item: object) -> None: ...


def env_worker(inq: MessageQueue, outq: MessageQueue) -> None:
    try:
        import io
        from contextlib import redirect_stderr, redirect_stdout

        import yaml
    except ImportError as err:
        raise ImportError(
            "alfworld is not installed. Please install it via `pip install alfworld==0.3.5` then run `alfworld-download`."
        ) from err
    buf = io.StringIO()
    config_path = Path(__file__).resolve().parent / "base_config.yml"
    environment = importlib.import_module("alfworld.agents.environment")
    with config_path.open() as f:
        config = yaml.safe_load(f)
    with redirect_stdout(buf), redirect_stderr(buf):
        base_env = environment.AlfredTWEnv(config, train_eval="train")
    env = None
    while True:
        message = inq.get()
        if not isinstance(message, tuple) or len(message) != 2:
            raise TypeError(f"Invalid message received by env worker: {message!r}")
        cmd, data = message
        if cmd == "init":
            env = base_env.init_env(batch_size=1)
            env.skip(data)
            task_def, info = env.reset()
            outq.put((task_def[0], info))
        elif cmd == "step":
            if env is None:
                outq.put("ENV_NOT_INITIALIZED")
                continue
            obs, rew, done, info = env.step([data])
            outq.put((obs, rew, done, info))
        elif cmd == "close":
            outq.put("CLOSED")
            break
        else:
            outq.put("UNKNOWN_CMD")


class EnvPool:
    def __init__(self, size: int = 2) -> None:
        self.size = size
        self.workers = []
        self.available = queue.Queue()
        try:
            mp = cast("Any", importlib.import_module("multiprocess"))
        except ImportError as err:
            raise ImportError(
                "multiprocess is not installed. Please install it via `pip install multiprocess`."
            ) from err
        with contextlib.suppress(RuntimeError):
            mp.set_start_method("spawn", force=True)
        ctx = mp.get_context("spawn")
        for i in range(size):
            inq = ctx.Queue()
            outq = ctx.Queue()
            p = ctx.Process(target=env_worker, args=(inq, outq), daemon=True)
            p.start()
            self.workers.append((inq, outq, p))
            self.available.put(i)

    def _acquire(self) -> tuple[int, MessageQueue, MessageQueue]:
        wid = self.available.get()
        return (wid, self.workers[wid][0], self.workers[wid][1])

    def _release(self, wid: int) -> None:
        self.available.put(wid)

    def close_all(self) -> None:
        while not self.available.empty():
            wid = self.available.get()
            inq, outq, proc = self.workers[wid]
            inq.put(("close", None))
            outq.get()
            inq.close()
            outq.close()
            proc.join()

    def session(self) -> _EnvSession:
        return _EnvSession(self)


class _EnvSession:
    def __init__(self, pool: EnvPool) -> None:
        self.pool = pool
        self.wid: int | None = None
        self.inq: MessageQueue | None = None
        self.outq: MessageQueue | None = None

    def __enter__(self) -> _EnvSession:
        self.wid, self.inq, self.outq = self.pool._acquire()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        if self.wid is not None:
            self.pool._release(self.wid)

    def init(self, idx: int) -> object:
        if self.inq is None or self.outq is None:
            raise RuntimeError("Session must be entered before calling init.")
        self.inq.put(("init", idx))
        return self.outq.get()

    def step(self, action: str) -> object:
        if self.inq is None or self.outq is None:
            raise RuntimeError("Session must be entered before calling step.")
        self.inq.put(("step", action))
        return self.outq.get()


class AlfWorld(Dataset):
    default_input_keys: ClassVar[tuple[str, ...]] = ("idx",)

    def __init__(
        self,
        max_threads: int = 20,
        train_seed: int = 0,
        train_size: int = 3000,
        dev_seed: int = 0,
        dev_size: int = 500,
        test_seed: int = 0,
        test_size: int | None = None,
        input_keys: list[str] | None = None,
    ) -> None:
        super().__init__(
            train_seed=train_seed,
            train_size=train_size,
            dev_seed=dev_seed,
            dev_size=dev_size,
            test_seed=test_seed,
            test_size=test_size,
            input_keys=input_keys or list(self.default_input_keys),
        )
        self.POOL = EnvPool(size=max_threads)
        records: list[dict[str, object]] = [{"idx": idx} for idx in range(3500)]
        random.Random(0).shuffle(records)
        self._train = records[:train_size]
        self._dev = records[-dev_size:]
        self._test = None

    def __del__(self) -> None:
        self.POOL.close_all()
