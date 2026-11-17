"""Member 4 experiment: SAOT versus the paper's w/o cross-task distillation."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_sequential import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/example.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    full = run(config, args.seed, args.smoke_test, no_distillation=False)
    without = run(config, args.seed, args.smoke_test, no_distillation=True)
    out = Path(config.get("output_root", "results")) / "figures" / "member_4"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "distillation_ablation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", "final_loss", "output"]); writer.writeheader()
        writer.writerow({"variant": "SAOT", "final_loss": full["final_loss"], "output": full["output"]})
        writer.writerow({"variant": "w/o Cross-task Distillation", "final_loss": without["final_loss"], "output": without["output"]})
    print(f"SAOT final loss: {full['final_loss']:.6f}")
    print(f"w/o distillation final loss: {without['final_loss']:.6f}")


if __name__ == "__main__":
    main()
