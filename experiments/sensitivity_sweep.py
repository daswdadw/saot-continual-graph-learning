"""Hyperparameter sweeps: alpha, beta, replay_budget (Member 6)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config_io import load_config, merge_config
from utils.paths import standard_run_dir
from utils.results_io import aggregate_runs
from utils.training_client import run_training

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SWEEP_PARAMS = ("alpha", "beta", "replay_budget")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "ablation.yaml")
    parser.add_argument("--param", choices=SWEEP_PARAMS, required=True)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    grid = base_cfg.get(f"{args.param}_grid")
    if not grid:
        raise SystemExit(f"Config missing {args.param}_grid list")

    rows = []
    dataset = str(base_cfg.get("dataset", "CoraFull-CL"))
    scenario = str(base_cfg.get("scenario", "class-il"))

    for val in grid:
        for seed in args.seeds:
            cfg = merge_config(base_cfg, {args.param: val, "seed": seed, "model": "saot", "variant": "SAOT (ours)"})
            run_name = f"sensitivity_{args.param}_{val}"
            out_dir = standard_run_dir(args.results_root, run_name, dataset, scenario, seed)
            metrics = run_training(cfg, out_dir)
            rows.append(
                {
                    args.param: val,
                    "seed": seed,
                    "AP": float(metrics["AP"]),
                    "AF": float(metrics["AF"]),
                    "dataset": dataset,
                    "scenario": scenario,
                    "run_dir": str(out_dir),
                }
            )

    out_dir = args.results_root / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    detail = out_dir / f"{args.param}_runs.csv"
    pd.DataFrame(rows).to_csv(detail, index=False)

    summary = aggregate_runs(rows, [args.param, "dataset", "scenario"])
    summary_csv = out_dir / f"{args.param}_summary.csv"
    summary.to_csv(summary_csv, index=False)

    # Backward-compatible single file for visualize.py
    slim = summary[[args.param, "AP_mean", "AF_mean"]].rename(columns={"AP_mean": "AP", "AF_mean": "AF"})
    slim["variant"] = "SAOT"
    slim["dataset"] = dataset
    slim["scenario"] = scenario
    slim.to_csv(out_dir / f"{args.param}_sweep.csv", index=False)

    logger.info("Wrote %s and %s", detail, summary_csv)


if __name__ == "__main__":
    main()
