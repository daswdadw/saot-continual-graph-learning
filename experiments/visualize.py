"""Generate all Member 6 figures from CSV/NPZ under results/ (Week 1 deliverable)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.plotting import plot_all_from_results_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--figures", type=Path, default=ROOT / "results" / "figures")
    args = parser.parse_args()
    plot_all_from_results_dir(args.results, args.figures)
    print(f"Figures saved to {args.figures}")


if __name__ == "__main__":
    main()
