"""
Compute task-shift sequence from class prototypes or Member 2 FGW JSON.

Outputs JSON consumed by adaptive_beta.py:

  {"task_shifts": [0.21, 0.35, ...], "source": "prototypes"|"fgw"}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.adaptive_beta import load_fgw_shifts, shifts_from_prototype_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build task_shifts.json for adaptive beta")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--prototype-dir", type=Path, help="Directory with task_XX.npz files")
    src.add_argument("--fgw-json", type=Path, help="FGW or task_shifts JSON from Member 2")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "adaptive_beta" / "task_shifts.json")
    args = parser.parse_args()

    if args.prototype_dir:
        shifts = shifts_from_prototype_dir(args.prototype_dir)
        payload = {"task_shifts": shifts, "source": "prototypes", "prototype_dir": str(args.prototype_dir)}
    else:
        shifts = load_fgw_shifts(args.fgw_json)
        payload = {"task_shifts": shifts, "source": "fgw", "fgw_json": str(args.fgw_json)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(shifts)} shifts to {args.out}")


if __name__ == "__main__":
    main()
