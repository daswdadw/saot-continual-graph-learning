"""Task-shift utilities for the adaptive cross-task distillation experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np


@dataclass(frozen=True)
class AdaptiveBetaConfig:
    beta_min: float = 0.3
    beta_max: float = 1.0
    beta_fixed: float = 0.6
    smoothing: float = 0.2
    metric: str = "prototype_cosine"

    def to_dict(self) -> dict:
        return asdict(self)


def prototype_cosine_shift(features_1: np.ndarray, labels_1: np.ndarray, features_2: np.ndarray, labels_2: np.ndarray) -> float:
    def prototypes(x, y):
        result = []
        for label in np.unique(y):
            row = x[y == label].mean(axis=0)
            norm = np.linalg.norm(row)
            result.append(row / max(norm, 1e-12))
        return np.asarray(result)
    p1, p2 = prototypes(np.asarray(features_1), np.asarray(labels_1)), prototypes(np.asarray(features_2), np.asarray(labels_2))
    if len(p1) == 0 or len(p2) == 0:
        return 0.0
    similarity = np.max(p1 @ p2.T, axis=1).mean()
    return float(np.clip(1.0 - similarity, 0.0, 1.0))


def compute_adaptive_beta(shift: float, config: AdaptiveBetaConfig) -> float:
    # A larger shift requires stronger consolidation in the proposed extension.
    return float(np.clip(config.beta_min + float(shift) * (config.beta_max - config.beta_min), config.beta_min, config.beta_max))


def betas_for_task_sequence(shifts: list[float], config: AdaptiveBetaConfig) -> list[float]:
    values = [config.beta_min]
    previous = config.beta_min
    for shift in shifts:
        target = compute_adaptive_beta(shift, config)
        previous = config.smoothing * previous + (1.0 - config.smoothing) * target
        values.append(float(np.clip(previous, config.beta_min, config.beta_max)))
    return values


def schedule_betas_for_tasks(shifts: list[float], config: AdaptiveBetaConfig) -> list[float]:
    return betas_for_task_sequence(shifts, config)[1:]


def shifts_from_prototype_dir(directory: str | Path) -> list[float]:
    files = sorted(Path(directory).glob("task_*.npz"))
    if len(files) < 2:
        raise ValueError("prototype directory must contain at least two task_XX.npz files")
    payload = [np.load(path) for path in files]
    return [prototype_cosine_shift(a["features"], a["labels"], b["features"], b["labels"]) for a, b in zip(payload, payload[1:])]


def load_fgw_shifts(path: str | Path) -> list[float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("task_shifts", "fgw_distances", "shifts"):
        if key in payload:
            return [float(value) for value in payload[key]]
    raise ValueError("shift JSON must contain task_shifts, fgw_distances, or shifts")
