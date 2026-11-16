"""Tests for the NumPy-only AP/AF metrics (no torch required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from evaluation.metrics import (
    PerformanceMatrix,
    aggregate_seeds,
    average_forgetting,
    average_performance,
    per_task_final_accuracy,
    plot_main_ap_af,
    plot_matrix_heatmap,
    summarize,
    write_rows_csv,
)

# Row i = accuracy measured after training task i (see the paper's Eq. for Mp).
REFERENCE = np.array(
    [
        [0.90, np.nan, np.nan],
        [0.80, 0.70, np.nan],
        [0.60, 0.50, 0.40],
    ]
)


def test_average_performance_is_mean_of_final_row() -> None:
    assert average_performance(REFERENCE) == pytest.approx(0.5)
    assert np.array_equal(per_task_final_accuracy(REFERENCE), np.array([0.6, 0.5, 0.4]))


def test_average_forgetting_matches_paper_formula() -> None:
    # AF = mean(M[T-1, i] - M[i, i]) = ((0.6-0.9) + (0.5-0.7)) / 2 = -0.25
    assert average_forgetting(REFERENCE) == pytest.approx(-0.25)


def test_single_task_cannot_be_forgotten() -> None:
    matrix = np.array([[0.42]])
    assert average_forgetting(matrix) == 0.0
    assert average_performance(matrix) == pytest.approx(0.42)


def test_partial_matrix_uses_the_evaluated_prefix() -> None:
    matrix = PerformanceMatrix(3)
    matrix.update_row(0, {0: 0.9})
    assert matrix.last_evaluated_row() == 0
    assert matrix.summary().ap == pytest.approx(0.9)
    matrix.update_row(1, {0: 0.4, 1: 0.8})
    assert matrix.summary().ap == pytest.approx(0.6)
    assert matrix.summary().af == pytest.approx(0.4 - 0.9)
    # The full matrix still has an unevaluated trailing row.
    assert np.isnan(average_performance(matrix.to_numpy()))


@pytest.mark.parametrize(
    "bad",
    [np.zeros((2, 3)), np.zeros(4), np.array([])],
    ids=["non-square", "one-dimensional", "empty"],
)
def test_invalid_matrices_raise(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        average_performance(bad)
    with pytest.raises(ValueError):
        summarize(bad)


def test_update_rejects_out_of_range_indices() -> None:
    matrix = PerformanceMatrix(2)
    with pytest.raises(IndexError):
        matrix.update(0, 2, 0.5)
    with pytest.raises(ValueError):
        matrix.update(0, 0, 1.5)


def test_csv_roundtrip_preserves_evaluated_cells(tmp_path: Path) -> None:
    matrix = PerformanceMatrix(3)
    for row_index, row in enumerate([[0.9], [0.8, 0.7], [0.6, 0.5, 0.4]]):
        matrix.update_row(row_index, {column: value for column, value in enumerate(row)})
    csv_path = matrix.to_csv(tmp_path / "matrix.csv")

    restored = PerformanceMatrix.from_csv(csv_path)
    lower = np.tril(np.ones((3, 3), dtype=bool))
    assert np.allclose(restored.to_numpy()[lower], REFERENCE[lower])
    assert np.isnan(restored.to_numpy()[np.triu_indices(3, k=1)]).all()
    assert restored.summary().ap == pytest.approx(0.5)


def test_aggregate_seeds_reports_percent_mean_and_std() -> None:
    rows = [
        {"model": "gae", "dataset": "d", "scenario": "class-il", "seed": 42, "ap": 0.5, "af": -0.25},
        {"model": "gae", "dataset": "d", "scenario": "class-il", "seed": 43, "ap": 0.7, "af": -0.05},
    ]
    aggregated = aggregate_seeds(rows)

    assert len(aggregated) == 1
    record = aggregated[0]
    assert record["num_seeds"] == 2
    assert record["seeds"] == "42;43"
    assert record["ap_mean"] == pytest.approx(60.0)
    assert record["ap_std"] == pytest.approx(np.std([0.5, 0.7], ddof=1) * 100.0)
    assert record["af_mean"] == pytest.approx(-15.0)


def test_write_rows_csv_keeps_declared_column_order(tmp_path: Path) -> None:
    path = write_rows_csv(
        tmp_path / "table.csv",
        [{"b": 2, "a": 1}],
        fieldnames=["a", "b"],
    )
    assert path.read_text(encoding="utf-8").splitlines()[0] == "a,b"


def test_plot_helpers_write_files(tmp_path: Path) -> None:
    rows = [
        {
            "model": "gae",
            "dataset": "d",
            "scenario": "class-il",
            "num_seeds": 2,
            "seeds": "42;43",
            "ap_mean": 60.0,
            "ap_std": 10.0,
            "af_mean": -15.0,
            "af_std": 2.0,
        }
    ]
    figure = plot_main_ap_af(rows, tmp_path / "ap_af.png")
    heatmap = plot_matrix_heatmap(REFERENCE, tmp_path / "matrix.png")
    if figure is None or heatmap is None:
        pytest.skip("matplotlib is not installed")
    assert figure.exists() and figure.stat().st_size > 0
    assert heatmap.exists() and heatmap.stat().st_size > 0
