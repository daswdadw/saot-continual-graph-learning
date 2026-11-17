"""Canonical result-directory naming."""

from __future__ import annotations

import re
from pathlib import Path


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
    return value.lower()


def standard_run_dir(results_root: Path, run_name: str, dataset: str, scenario: str, seed: int) -> Path:
    return Path(results_root) / slugify(run_name) / slugify(dataset) / str(scenario).lower() / f"seed_{int(seed)}"


def ablation_run_dir(results_root: Path, variant: str, seed: int) -> Path:
    return Path(results_root) / "ablation" / slugify(variant) / f"seed_{int(seed)}"
