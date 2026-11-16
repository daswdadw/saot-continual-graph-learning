"""Unified continual-graph-learning evaluation metrics.

This module deliberately has no PyTorch dependency, so AP/AF can always be
recomputed from a saved ``performance_matrix.csv``::

    python -c "from evaluation.metrics import PerformanceMatrix, summarize; \
print(summarize(PerformanceMatrix.from_csv('results/.../performance_matrix.csv')))

Definitions follow the SAOT paper (Section 5.3):

* ``M[i, j]`` is the accuracy on task ``j`` after the model finished training
  task ``i`` (0-based).  Cells that were never evaluated stay ``NaN``.
* ``AP = mean(M[T-1, :])``  -- average performance (final row).
* ``AF = mean(M[T-1, i] - M[i, i])`` for ``i < T-1`` -- average forgetting;
  negative means forgetting, positive means backward transfer.  ``T == 1``
  yields ``AF = 0.0`` because nothing can be forgotten yet.

Values are accuracy fractions in ``[0, 1]``.  CSV tables and plots use
percentages to match the paper.
"""

from __future__ import annotations

import csv
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "MatrixSummary",
    "PerformanceMatrix",
    "accuracy_to_percent",
    "aggregate_seeds",
    "average_forgetting",
    "average_performance",
    "per_task_final_accuracy",
    "plot_main_ap_af",
    "plot_matrix_heatmap",
    "summarize",
    "write_rows_csv",
]

_UNEVALUATED_CELLS = {"", "nan", "NaN", "None", "null"}
_CSV_INDEX_COLUMN = "trained_until"


def accuracy_to_percent(value: float) -> float:
    """Convert an accuracy fraction to a percentage."""

    return float(value) * 100.0


def _as_matrix(matrix: Any) -> np.ndarray:
    if isinstance(matrix, PerformanceMatrix):
        return matrix.to_numpy(copy=False)
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2:
        raise ValueError(
            f"performance matrix must be 2-D [T, T], got {array.ndim} dimension(s)."
        )
    if array.size == 0:
        raise ValueError("performance matrix must not be empty.")
    if array.shape[0] != array.shape[1]:
        raise ValueError(
            f"performance matrix must be square, got shape {array.shape}."
        )
    return array


def average_performance(matrix: Any) -> float:
    """Return ``AP``: the mean accuracy of the last row.

    Cells that are ``NaN`` (task not evaluated yet) are ignored.  A last row
    without any evaluated task returns ``NaN``.
    """

    array = _as_matrix(matrix)
    last_row = array[-1]
    valid = last_row[~np.isnan(last_row)]
    if valid.size == 0:
        return float("nan")
    return float(valid.mean())


def average_forgetting(matrix: Any) -> float:
    """Return ``AF``: mean accuracy drop relative to right after each task.

    ``AF = mean(M[T-1, i] - M[i, i])`` over ``i in [0, T-2]``.  This is the
    paper definition; it is *not* the ``max`` over past rows used by some CGL
    implementations.  ``T == 1`` returns ``0.0``.
    """

    array = _as_matrix(matrix)
    num_tasks = array.shape[0]
    if num_tasks < 2:
        return 0.0
    final_row = array[-1, : num_tasks - 1]
    diagonal = np.diagonal(array)[: num_tasks - 1]
    valid = ~(np.isnan(final_row) | np.isnan(diagonal))
    if not valid.any():
        return float("nan")
    return float((final_row[valid] - diagonal[valid]).mean())


def per_task_final_accuracy(matrix: Any) -> np.ndarray:
    """Return the last row of the performance matrix (per-task final accuracy)."""

    return _as_matrix(matrix)[-1].copy()


@dataclass(frozen=True)
class MatrixSummary:
    """AP/AF summary of one performance matrix."""

    num_tasks: int
    ap: float
    af: float
    per_task_final: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_tasks": self.num_tasks,
            "ap": self.ap,
            "af": self.af,
            "ap_percent": accuracy_to_percent(self.ap)
            if not math.isnan(self.ap)
            else float("nan"),
            "af_percent": accuracy_to_percent(self.af)
            if not math.isnan(self.af)
            else float("nan"),
            "per_task_final_accuracy": [
                None if math.isnan(value) else float(value) for value in self.per_task_final
            ],
        }


def summarize(matrix: Any) -> MatrixSummary:
    """Return the AP/AF summary of ``matrix``."""

    array = _as_matrix(matrix)
    return MatrixSummary(
        num_tasks=array.shape[0],
        ap=average_performance(array),
        af=average_forgetting(array),
        per_task_final=per_task_final_accuracy(array),
    )


class PerformanceMatrix:
    """``T x T`` performance matrix with ``NaN`` for not-yet-evaluated cells.

    Row ``i`` holds the accuracies measured after training task ``i``; column
    ``j`` is the task being evaluated.
    """

    def __init__(self, num_tasks: int) -> None:
        if isinstance(num_tasks, bool) or not isinstance(num_tasks, int):
            raise TypeError(f"num_tasks must be an integer, got {num_tasks!r}.")
        if num_tasks < 1:
            raise ValueError(f"num_tasks must be positive, got {num_tasks}.")
        self._data = np.full((num_tasks, num_tasks), np.nan, dtype=float)

    @property
    def num_tasks(self) -> int:
        return int(self._data.shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.num_tasks, self.num_tasks)

    def __len__(self) -> int:
        return self.num_tasks

    def __repr__(self) -> str:
        return f"PerformanceMatrix(num_tasks={self.num_tasks})"

    def update(self, trained_until: int, task_index: int, accuracy: float) -> None:
        """Record the accuracy on ``task_index`` after training ``trained_until``."""

        self._validate_index("trained_until", trained_until)
        self._validate_index("task_index", task_index)
        value = float(accuracy)
        if not math.isnan(value) and not 0.0 <= value <= 1.0:
            raise ValueError(f"accuracy must be in [0, 1] or NaN, got {accuracy!r}.")
        self._data[trained_until, task_index] = value

    def update_row(self, trained_until: int, accuracies: Mapping[int, float] | Sequence[float]) -> None:
        """Record a whole evaluated row at once."""

        if isinstance(accuracies, Mapping):
            items = accuracies.items()
        else:
            items = enumerate(accuracies)
        for task_index, accuracy in items:
            self.update(trained_until, int(task_index), accuracy)

    def last_row(self) -> np.ndarray:
        return self._data[-1].copy()

    def last_evaluated_row(self) -> int | None:
        """Index of the last row containing at least one evaluated cell."""

        filled = ~np.isnan(self._data).all(axis=1)
        indices = np.flatnonzero(filled)
        return int(indices[-1]) if indices.size else None

    def evaluated_prefix(self) -> np.ndarray:
        """Return the square block up to the last evaluated row.

        Useful while an experiment is still running: AP/AF then describe the
        tasks finished so far instead of a trailing row of ``NaN``.
        """

        last = self.last_evaluated_row()
        if last is None:
            return np.empty((0, 0), dtype=float)
        return self._data[: last + 1, : last + 1].copy()

    def to_numpy(self, copy: bool = True) -> np.ndarray:
        return self._data.copy() if copy else self._data

    def summary(self, *, prefix: bool = True) -> MatrixSummary:
        """Summarise the matrix; ``prefix=False`` always uses all tasks."""

        return summarize(self.evaluated_prefix() if prefix else self._data)

    def to_csv(self, path: str | Path) -> Path:
        """Write the matrix; unevaluated cells are written as empty strings."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        header = [_CSV_INDEX_COLUMN] + [f"task_{index}" for index in range(self.num_tasks)]
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for row_index in range(self.num_tasks):
                row = [str(row_index)]
                for value in self._data[row_index]:
                    row.append("" if np.isnan(value) else f"{float(value):.6f}")
                writer.writerow(row)
        return target

    @classmethod
    def from_csv(cls, path: str | Path) -> "PerformanceMatrix":
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"performance matrix not found: {source}")
        with source.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            task_columns = [name for name in fieldnames if name.startswith("task_")]
            if not task_columns:
                raise ValueError(f"{source} has no task_* columns.")
            matrix = cls(len(task_columns))
            for csv_row in reader:
                row_index = int(csv_row[_CSV_INDEX_COLUMN])
                for column_index, column in enumerate(task_columns):
                    raw = (csv_row.get(column) or "").strip()
                    if raw in _UNEVALUATED_CELLS:
                        continue
                    matrix.update(row_index, column_index, float(raw))
        return matrix

    def _validate_index(self, name: str, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer, got {value!r}.")
        if not 0 <= value < self.num_tasks:
            raise IndexError(
                f"{name}={value} is out of range for {self.num_tasks} tasks."
            )


def _finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray([float(value) for value in values], dtype=float)
    return array[~np.isnan(array)]


def aggregate_seeds(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_keys: Sequence[str] = ("model", "dataset", "scenario"),
    value_keys: Sequence[str] = ("ap", "af"),
    percent: bool = True,
) -> list[dict[str, Any]]:
    """Aggregate per-seed rows into ``mean``/``std`` rows (percent by default)."""

    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row.get(name, "")) for name in group_keys)
        groups.setdefault(key, []).append(row)

    scale = 100.0 if percent else 1.0
    aggregated: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = groups[key]
        record: dict[str, Any] = dict(zip(group_keys, key))
        seeds = sorted({int(row["seed"]) for row in group if row.get("seed") is not None})
        record["num_seeds"] = len(seeds)
        record["seeds"] = ";".join(str(seed) for seed in seeds)
        for value_key in value_keys:
            values = _finite(
                row[value_key] for row in group if row.get(value_key) is not None
            )
            if values.size == 0:
                mean = float("nan")
                std = float("nan")
            else:
                mean = float(values.mean())
                std = float(values.std(ddof=1)) if values.size > 1 else 0.0
            record[f"{value_key}_mean"] = mean * scale
            record[f"{value_key}_std"] = std * scale
        aggregated.append(record)
    return aggregated


def write_rows_csv(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> Path:
    """Write homogeneous rows to CSV, creating parent directories."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if fieldnames:
            with target.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(list(fieldnames))
        return target
    columns = list(fieldnames) if fieldnames else list(rows[0].keys())
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in columns})
    return target


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
        from matplotlib import pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib is optional at runtime
        warnings.warn(
            "matplotlib is not installed; skipping figure output.", RuntimeWarning
        )
        return None
    return plt


def plot_main_ap_af(
    rows: Sequence[Mapping[str, Any]],
    path: str | Path,
    *,
    title: str = "AP / AF (mean +- std over seeds)",
    scenarios: Sequence[str] = ("class-il", "task-il"),
    models: Sequence[str] | None = None,
) -> Path | None:
    """Plot a 2-column AP/AF bar chart with one row per scenario.

    ``rows`` are :func:`aggregate_seeds` outputs (percent values).  Returns the
    written path, or ``None`` when matplotlib is unavailable.
    """

    plt = _pyplot()
    if plt is None:
        return None

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        nrows=len(scenarios),
        ncols=2,
        figsize=(11.5, 3.8 * max(1, len(scenarios))),
        squeeze=False,
    )
    colours = plt.get_cmap("tab10").colors

    for row_index, scenario in enumerate(scenarios):
        subset = [
            row
            for row in rows
            if str(row.get("scenario")) == scenario
            and (models is None or str(row.get("model")) in set(models))
        ]
        subset.sort(key=lambda row: str(row.get("model")))
        names = [str(row.get("model")) for row in subset]

        for column_index, metric in enumerate(("ap", "af")):
            axis = axes[row_index][column_index]
            axis.set_title(f"{scenario.upper()} — {metric.upper()}")
            axis.set_ylabel(f"{metric.upper()} (%)")
            axis.grid(axis="y", alpha=0.3)
            if metric == "af":
                axis.axhline(0.0, color="black", linewidth=0.8)
            if not subset:
                axis.text(
                    0.5,
                    0.5,
                    "no runs",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
                continue
            means = [float(row.get(f"{metric}_mean", float("nan"))) for row in subset]
            stds = [float(row.get(f"{metric}_std", 0.0)) for row in subset]
            positions = list(range(len(names)))
            axis.bar(
                positions,
                means,
                yerr=stds,
                capsize=4,
                color=[colours[index % len(colours)] for index in positions],
                alpha=0.85,
            )
            axis.set_xticks(positions)
            axis.set_xticklabels(names, rotation=15)
            for position, mean, std in zip(positions, means, stds):
                if math.isnan(mean):
                    continue
                axis.annotate(
                    f"{mean:.1f}",
                    (position, mean),
                    textcoords="offset points",
                    xytext=(0, 6 if mean >= 0 else -12),
                    ha="center",
                    fontsize=8,
                )

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(target, dpi=200)
    plt.close(fig)
    return target


def plot_matrix_heatmap(
    matrix: Any,
    path: str | Path,
    *,
    title: str = "performance matrix",
    annotate: bool = True,
) -> Path | None:
    """Plot one performance matrix as a heatmap (percent accuracy, NaN in grey)."""

    plt = _pyplot()
    if plt is None:
        return None

    array = _as_matrix(matrix)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.4, 5.4))
    masked = np.ma.masked_invalid(array * 100.0)
    colormap = plt.get_cmap("viridis").copy()
    colormap.set_bad("#d9d9d9")
    image = axis.imshow(masked, cmap=colormap, vmin=0.0, vmax=100.0)
    axis.set_xlabel("evaluated task")
    axis.set_ylabel("trained until task")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="accuracy (%)")
    if annotate and array.shape[0] <= 40:
        for row_index in range(array.shape[0]):
            for column_index in range(row_index + 1):
                value = array[row_index, column_index]
                if math.isnan(value):
                    continue
                axis.text(
                    column_index,
                    row_index,
                    f"{value * 100.0:.0f}",
                    ha="center",
                    va="center",
                    fontsize=4.5,
                    color="white" if value < 0.6 else "black",
                )
    figure.tight_layout()
    figure.savefig(target, dpi=200)
    plt.close(figure)
    return target
