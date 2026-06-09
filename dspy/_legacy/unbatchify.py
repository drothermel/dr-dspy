"""Legacy convenience types; prefer typed models for new code.

Background batching helper for sync callables. Callers must invoke ``close()``
or use the context manager when done.
"""

import queue
import threading
import time
from concurrent.futures import Future
from types import TracebackType
from typing import Callable, Generic, TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class Unbatchify(Generic[InputT, OutputT]):
    def __init__(
        self, batch_fn: Callable[[list[InputT]], list[OutputT]], max_batch_size: int = 32, max_wait_time: float = 0.1
    ) -> None:
        self.batch_fn = batch_fn
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.input_queue: queue.Queue[tuple[InputT, Future[OutputT]]] = queue.Queue()
        self.stop_event = threading.Event()
        self._closed = False
        self.worker_thread = threading.Thread(target=self._worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()

    def __call__(self, input_item: InputT) -> OutputT:
        future: Future[OutputT] = Future()
        self.input_queue.put((input_item, future))
        return future.result()

    def _worker(self) -> None:
        while not self.stop_event.is_set():
            batch: list[InputT] = []
            futures: list[Future[OutputT]] = []
            start_time = time.time()
            while len(batch) < self.max_batch_size and time.time() - start_time < self.max_wait_time:
                try:
                    input_item, future = self.input_queue.get(timeout=self.max_wait_time)
                    batch.append(input_item)
                    futures.append(future)
                except queue.Empty:
                    break
            if batch:
                try:
                    outputs = self.batch_fn(batch)
                    for output, future in zip(outputs, futures, strict=False):
                        future.set_result(output)
                except Exception as e:
                    for future in futures:
                        future.set_exception(e)
            else:
                time.sleep(0.01)
        while True:
            try:
                _, future = self.input_queue.get_nowait()
                future.set_exception(RuntimeError("Unbatchify is closed"))
            except queue.Empty:
                break

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self.stop_event.is_set():
            self.stop_event.set()
            self.worker_thread.join()

    def __enter__(self) -> "Unbatchify[InputT, OutputT]":
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.close()
