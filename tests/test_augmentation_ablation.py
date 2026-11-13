"""Tests for the member 3 code-level ablation runner."""

import csv
from pathlib import Path

from experiments.augmentation_ablation import run_augmentation_ablation


def test_ablation_runner_writes_reproducible_csv_outputs(tmp_path: Path) -> None:
    outputs = run_augmentation_ablation(
        {"feature_mask": 0.2, "hidden_dim": 8, "num_tasks": 3},
        seed=42,
        output_dir=tmp_path,
        edge_drop_values=[0.0, 0.5],
        buffer_budgets=[0, 8],
        num_nodes=32,
        input_dim=6,
        device="cpu",
    )
    for path in outputs.values():
        assert path.exists()

    with outputs["augmentation_csv"].open(newline="", encoding="utf-8") as handle:
        augmentation_rows = list(csv.DictReader(handle))
    with outputs["buffer_csv"].open(newline="", encoding="utf-8") as handle:
        buffer_rows = list(csv.DictReader(handle))
    assert len(augmentation_rows) == 2
    assert len(buffer_rows) == 2
    assert int(buffer_rows[0]["buffer_size"]) == 0
    assert int(buffer_rows[0]["loader_batches"]) == 0
    assert int(buffer_rows[1]["buffer_size"]) <= 8
