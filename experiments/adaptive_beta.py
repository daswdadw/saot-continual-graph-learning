"""Fixed vs adaptive beta comparison (Member 6 extension)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.adaptive_beta import AdaptiveBetaConfig, betas_for_task_sequence, load_fgw_shifts, shifts_from_prototype_dir
from utils.config_io import load_config, merge_config, save_config
from utils.results_io import aggregate_runs
from utils.training_client import run_training

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def resolve_shifts(args: argparse.Namespace, cfg: dict) -> list[float]:
    if args.shifts_json:
        return load_fgw_shifts(args.shifts_json)
    if args.prototype_dir:
        return shifts_from_prototype_dir(args.prototype_dir)
    raise SystemExit("Provide --shifts-json or --prototype-dir (run compute_task_shifts.py first).")


def run_mode(
    base_cfg: dict,
    mode: str,
    betas: list[float],
    seed: int,
    out_root: Path,
    ab_cfg: AdaptiveBetaConfig,
) -> dict:
    cfg = merge_config(base_cfg, {"model": "saot", "seed": seed, "beta_mode": mode})
    if mode == "fixed":
        cfg["beta"] = ab_cfg.beta_fixed
        cfg["beta_schedule"] = [ab_cfg.beta_fixed] * len(betas)
    else:
        cfg["beta_schedule"] = betas
        cfg.pop("beta", None)

    run_dir = out_root / mode / f"seed_{seed}"
    save_config({"betas": betas, **ab_cfg.to_dict()}, run_dir / "beta_schedule_meta.yaml")
    (run_dir / "beta_schedule.json").write_text(json.dumps(betas, indent=2), encoding="utf-8")

    metrics = run_training(cfg, run_dir)
    return {
        "mode": mode,
        "seed": seed,
        "AP": float(metrics["AP"]),
        "AF": float(metrics["AF"]),
        "dataset": cfg.get("dataset"),
        "scenario": cfg.get("scenario"),
        "run_dir": str(run_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive beta vs fixed beta")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "adaptive_beta.yaml")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results" / "adaptive_beta")
    parser.add_argument("--shifts-json", type=Path, default=None)
    parser.add_argument("--prototype-dir", type=Path, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    args = parser.parse_args()

    cfg = load_config(args.config)
    ab_cfg = AdaptiveBetaConfig(
        beta_min=float(cfg.get("beta_min", 0.3)),
        beta_max=float(cfg.get("beta_max", 1.0)),
        beta_fixed=float(cfg.get("beta_fixed", 0.6)),
        smoothing=float(cfg.get("beta_smoothing", 0.2)),
        metric=str(cfg.get("shift_metric", "prototype_cosine")),
    )

    shifts = resolve_shifts(args, cfg)
    betas = betas_for_task_sequence(shifts, ab_cfg)
    num_tasks = int(cfg.get("num_tasks", len(betas)))
    if len(betas) != num_tasks:
        logger.warning("num_tasks=%s but beta length=%s; truncating/padding", num_tasks, len(betas))
        if len(betas) > num_tasks:
            betas = betas[:num_tasks]
        else:
            betas = betas + [betas[-1]] * (num_tasks - len(betas))

    args.results_root.mkdir(parents=True, exist_ok=True)
    meta_path = args.results_root / "schedule_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "shifts": shifts,
                "betas_adaptive": betas,
                "rule": "beta_t: task0=beta_min; later tasks from shift with EMA",
                **ab_cfg.to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    rows = []
    for seed in args.seeds:
        rows.append(run_mode(cfg, "fixed", betas, seed, args.results_root, ab_cfg))
        rows.append(run_mode(cfg, "adaptive", betas, seed, args.results_root, ab_cfg))

    df = pd.DataFrame(rows)
    df.to_csv(args.results_root / "fixed_vs_adaptive_runs.csv", index=False)
    summary = aggregate_runs(rows, ["mode", "dataset", "scenario"])
    summary.to_csv(args.results_root / "fixed_vs_adaptive_summary.csv", index=False)
    logger.info("Done. Summary:\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
