"""Adapter from experiment scripts to the integrated sequential trainer."""

from __future__ import annotations

from pathlib import Path
import json


def run_training(config: dict, output_dir: str | Path) -> dict[str, float]:
    if not config.get("task_file"):
        raise ValueError("config.task_file is required for integrated SAOT training")
    from experiments.train_sequential import run
    run_config = dict(config)
    run_config["output_root"] = str(Path(output_dir).parent.parent.parent.parent)
    result = run(run_config, int(config.get("seed", 42)), smoke_test=False, no_distillation=not config.get("use_cross_task_distillation", True))
    return {"AP": float(result["average_performance"]), "AF": float(result["average_forgetting"])}
