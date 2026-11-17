"""Continual-learning metrics from the SAOT paper Section 5.3."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np


def average_performance(performance_matrix: np.ndarray) -> float:
    matrix = np.asarray(performance_matrix, dtype=float)
    return float(np.nanmean(matrix[-1]))


def average_forgetting(performance_matrix: np.ndarray) -> float:
    matrix = np.asarray(performance_matrix, dtype=float)
    if matrix.shape[0] < 2:
        return 0.0
    t = min(matrix.shape)
    return float(np.nanmean([matrix[-1, i] - matrix[i, i] for i in range(t - 1)]))


def save_performance_matrix(matrix: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(matrix, dtype=float), delimiter=",", fmt="%.8f")
