"""Minimal result plotting entry point used by experiments/visualize.py."""

from __future__ import annotations

from pathlib import Path
import csv

import matplotlib.pyplot as plt


def plot_all_from_results_dir(results_dir: str | Path, figures_dir: str | Path) -> None:
    results_dir, figures_dir = Path(results_dir), Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    for csv_path in results_dir.rglob("*.csv"):
        if "performance_matrix" not in csv_path.name:
            continue
        rows = list(csv.reader(csv_path.open(encoding="utf-8-sig")))
        if len(rows) < 2:
            continue
        values = []
        for row in rows[1:]:
            values.append([float(x) if x else float("nan") for x in row[1:]])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(values, aspect="auto", interpolation="nearest")
        ax.set_title(csv_path.parent.name)
        ax.set_xlabel("Evaluation task")
        ax.set_ylabel("After task")
        fig.tight_layout()
        fig.savefig(figures_dir / f"{csv_path.parent.name}_performance_matrix.png", dpi=160)
        plt.close(fig)
