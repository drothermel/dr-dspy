import asyncio
import csv
import importlib
import importlib.util
import json
import logging
import types
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from typing_extensions import override

if TYPE_CHECKING:
    import pandas as pd

    from dspy.primitives.example import Example
    from dspy.primitives.module import Module
import tqdm

from dspy.primitives.prediction import Prediction
from dspy.runtime.run_context import RunContext
from dspy.utils.async_parallel import run_bounded
from dspy.utils.callback import with_callbacks

logger = logging.getLogger(__name__)


class EvaluationResult(Prediction):
    def __init__(self, score: float, results: list[tuple["Example", "Example", Any]]) -> None:
        super().__init__(score=score, results=results)

    @override
    def __repr__(self) -> str:
        return f"EvaluationResult(score={self.score}, results=<list of {len(self.results)} results>)"


class Evaluate:
    def __init__(
        self,
        *,
        devset: list["Example"],
        metric: Callable | None = None,
        num_threads: int | None = None,
        max_concurrency: int | None = None,
        display_progress: bool = False,
        display_table: bool | int = False,
        max_errors: int | None = None,
        provide_traceback: bool | None = None,
        failure_score: float = 0.0,
        save_as_csv: str | None = None,
        save_as_json: str | None = None,
    ) -> None:
        self.devset = devset
        self.metric = metric
        self.max_concurrency = max_concurrency if max_concurrency is not None else num_threads
        self.num_threads = self.max_concurrency
        self.display_progress = display_progress
        self.display_table = display_table
        self.max_errors = max_errors
        self.provide_traceback = provide_traceback
        self.failure_score = failure_score
        self.save_as_csv = save_as_csv
        self.save_as_json = save_as_json

    @with_callbacks(kind="evaluate")
    async def __call__(
        self,
        program: "Module",
        run: RunContext,
        metric: Callable | None = None,
        devset: list["Example"] | None = None,
        num_threads: int | None = None,
        max_concurrency: int | None = None,
        display_progress: bool | None = None,
        display_table: bool | int | None = None,
        callback_metadata: dict[str, Any] | None = None,
        save_as_csv: str | None = None,
        save_as_json: str | None = None,
    ) -> EvaluationResult:
        metric = metric if metric is not None else self.metric
        devset = devset if devset is not None else self.devset
        concurrency = (
            max_concurrency
            if max_concurrency is not None
            else num_threads
            if num_threads is not None
            else self.max_concurrency
        )
        display_progress = display_progress if display_progress is not None else self.display_progress
        display_table = display_table if display_table is not None else self.display_table
        save_as_csv = save_as_csv if save_as_csv is not None else self.save_as_csv
        save_as_json = save_as_json if save_as_json is not None else self.save_as_json
        if callback_metadata:
            logger.debug(f"Evaluate is called with callback metadata: {callback_metadata}")
        if metric is None:
            raise ValueError("A metric function is required for evaluation.")
        tqdm.tqdm._instances.clear()

        async def process_item(example):
            item_run = run.fork(trace=[])
            prediction = await program(**example.as_inputs(), run=item_run)
            trace = list(item_run.trace)
            score = metric(example, prediction, trace)
            return (prediction, score)

        max_errors = self.max_errors if self.max_errors is not None else run.execution.max_errors
        provide_traceback = (
            self.provide_traceback if self.provide_traceback is not None else run.execution.provide_traceback
        )
        results, _stats = await run_bounded(
            items=devset,
            fn=process_item,
            max_concurrency=concurrency or run.execution.num_threads,
            disable_progress_bar=not display_progress,
            max_errors=max_errors,
            provide_traceback=provide_traceback,
            compare_results=True,
        )
        assert len(devset) == len(results)
        results = [(Prediction(), self.failure_score) if r is None else r for r in results]
        results = [(example, prediction, score) for example, (prediction, score) in zip(devset, results, strict=False)]
        ncorrect, ntotal = (sum((score for *_, score in results)), len(devset))
        logger.info(f"Average Metric: {ncorrect} / {ntotal} ({round(100 * ncorrect / ntotal, 1)}%)")
        if display_table:
            if importlib.util.find_spec("pandas") is not None:
                metric_name = metric.__name__ if isinstance(metric, types.FunctionType) else metric.__class__.__name__
                result_df = self._construct_result_table(results, metric_name)
                self._display_result_table(result_df, display_table, metric_name)
            else:
                logger.warning("Skipping table display since `pandas` is not installed.")
        if save_as_csv:
            metric_name = metric.__name__ if isinstance(metric, types.FunctionType) else metric.__class__.__name__
            data = self._prepare_results_output(results, metric_name)
            await asyncio.to_thread(self._write_results_csv, save_as_csv, data)
        if save_as_json:
            metric_name = metric.__name__ if isinstance(metric, types.FunctionType) else metric.__class__.__name__
            data = self._prepare_results_output(results, metric_name)
            await asyncio.to_thread(self._write_results_json, save_as_json, data)
        return EvaluationResult(score=round(100 * ncorrect / ntotal, 2), results=results)

    @staticmethod
    def _prepare_results_output(results: list[tuple["Example", "Example", Any]], metric_name: str):
        return [
            merge_dicts(example, prediction) | {metric_name: score}
            if prediction_is_dictlike(prediction)
            else example.to_dict() | {"prediction": prediction, metric_name: score}
            for example, prediction, score in results
        ]

    @staticmethod
    def _write_results_csv(path: str, data: list[dict[str, Any]]) -> None:
        with Path(path).open("w", newline="") as csvfile:
            fieldnames = data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow(row)

    @staticmethod
    def _write_results_json(path: str, data: list[dict[str, Any]]) -> None:
        with Path(path).open("w") as f:
            json.dump(data, f)

    def _construct_result_table(
        self, results: list[tuple["Example", "Example", Any]], metric_name: str
    ) -> "pd.DataFrame":
        import pandas as pd

        data = self._prepare_results_output(results, metric_name)
        result_df = pd.DataFrame(data)
        result_df = result_df.map(truncate_cell) if hasattr(result_df, "map") else result_df.applymap(truncate_cell)
        return result_df.rename(columns={"correct": metric_name})

    def _display_result_table(self, result_df: "pd.DataFrame", display_table: bool | int, metric_name: str) -> None:
        if isinstance(display_table, bool):
            df_to_display = result_df.copy()
            truncated_rows = 0
        else:
            df_to_display = result_df.head(display_table).copy()
            truncated_rows = len(result_df) - display_table
        df_to_display = stylize_metric_name(df_to_display, metric_name)
        display_dataframe(df_to_display)
        if truncated_rows > 0:
            logger.info("%s more rows not displayed", truncated_rows)


def prediction_is_dictlike(prediction):
    return hasattr(prediction, "items") and callable(prediction.items)


def merge_dicts(d1, d2) -> dict:
    if hasattr(d1, "to_dict"):
        d1 = d1.to_dict()
    if hasattr(d2, "to_dict"):
        d2 = d2.to_dict()
    merged = {}
    for k, v in d1.items():
        if k in d2:
            merged[f"example_{k}"] = v
        else:
            merged[k] = v
    for k, v in d2.items():
        if k in d1:
            merged[f"pred_{k}"] = v
        else:
            merged[k] = v
    return merged


def truncate_cell(content) -> str:
    words = str(content).split()
    if len(words) > 25:
        return " ".join(words[:25]) + "..."
    return content


def stylize_metric_name(df: "pd.DataFrame", metric_name: str) -> "pd.DataFrame":

    def format_metric(x) -> str:
        if isinstance(x, float):
            return f"✔️ [{x:.3f}]"
        if x is not None:
            return f"✔️ [{x}]"
        return ""

    df[metric_name] = df[metric_name].apply(format_metric)
    return df


def display_dataframe(df: "pd.DataFrame") -> None:
    import pandas as pd

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", 70):
        pass
