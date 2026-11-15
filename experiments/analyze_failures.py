"""
Identify failure cases from performance matrices (Member 6).

For each run, finds tasks with largest forgetting after full training:
  drop_i = M[T-1, i] - M[i, i]   (same as AF term, per task)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def per_task_forgetting(matrix: pd.DataFrame) -> pd.DataFrame:
    arr = matrix.values
    t = arr.shape[0]
    if t < 2:
        raise ValueError("Performance matrix must be at least 2x2")
    rows = []
    for i in range(t - 1):
        drop = float(arr[t - 1, i] - arr[i, i])
        rows.append({"task_index": i, "acc_after_own_task": float(arr[i, i]), "acc_final": float(arr[t - 1, i]), "forgetting": drop})
    return pd.DataFrame(rows).sort_values("forgetting")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory with performance_matrix.csv")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    pm_path = args.run_dir / "performance_matrix.csv"
    df = pd.read_csv(pm_path, index_col=0)
    ranked = per_task_forgetting(df)
    worst = ranked.head(args.top_k).to_dict(orient="records")

    payload = {"run_dir": str(args.run_dir), "worst_tasks": worst, "all_tasks": ranked.to_dict(orient="records")}
    out = args.out or (args.run_dir / "failure_analysis.json")
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(worst, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
