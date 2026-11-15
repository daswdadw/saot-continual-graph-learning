"""Run SAOT ablation variants (Member 6)."""

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
from utils.paths import ablation_run_dir, slugify
from utils.results_io import aggregate_runs
from utils.training_client import run_training
from utils.variants import ABLATION_VARIANTS, validate_single_component_change, variant_to_overrides

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_ablation_variant(
    base_cfg: dict,
    variant: str,
    seed: int,
    results_root: Path,
) -> dict:
    validate_single_component_change(variant)
    overrides = variant_to_overrides(variant)
    cfg = merge_config(base_cfg, overrides)
    cfg["seed"] = seed
    cfg["variant"] = variant
    cfg["model"] = "saot"

    out_dir = ablation_run_dir(results_root, variant, seed)
    metrics = run_training(cfg, out_dir)
    return {
        "dataset": cfg.get("dataset"),
        "scenario": cfg.get("scenario"),
        "variant": variant,
        "seed": seed,
        "AP": float(metrics["AP"]),
        "AF": float(metrics["AF"]),
        "run_dir": str(out_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SAOT ablation on CoraFull-CL (Member 6)")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "ablation.yaml")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--variants", nargs="*", default=list(ABLATION_VARIANTS))
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    scenario = str(base_cfg.get("scenario", "class-il")).lower()
    dataset = str(base_cfg.get("dataset", "CoraFull-CL"))

    rows = []
    for variant in args.variants:
        logger.info("Variant: %s", variant)
        for seed in args.seeds:
            rows.append(run_ablation_variant(base_cfg, variant, seed, args.results_root))

    df = pd.DataFrame(rows)
    ablation_dir = args.results_root / "ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)

    slug = slugify(dataset)
    tag = "class_il" if scenario == "class-il" else "task_il"
    detail_csv = ablation_dir / f"{slug}_{tag}_ablation_runs.csv"
    df.to_csv(detail_csv, index=False)

    summary = aggregate_runs(rows, ["variant", "dataset", "scenario"])
    summary_csv = ablation_dir / f"{slug}_{tag}_ablation_summary.csv"
    summary.to_csv(summary_csv, index=False)

    logger.info("Wrote %s", detail_csv)
    logger.info("Wrote %s", summary_csv)


if __name__ == "__main__":
    main()
