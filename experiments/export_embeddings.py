"""
Build side-by-side embedding NPZ for visualization (Member 6).

Reads two completed runs (baseline vs SAOT) and writes:

  results/embeddings/<name>.npz
    baseline_emb, saot_emb, labels
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_embeddings(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    npz_path = run_dir / "embeddings.npz"
    if npz_path.exists():
        data = np.load(npz_path)
        key = "emb" if "emb" in data else "embeddings"
        if key not in data:
            raise KeyError(f"{npz_path} missing 'emb' or 'embeddings'")
        return data[key], data["labels"]

    pt_path = run_dir / "checkpoint.pt"
    if not pt_path.exists():
        raise FileNotFoundError(
            f"No embeddings.npz or checkpoint.pt in {run_dir}. "
            "Ask Member 4 to export embeddings.npz after the last task."
        )

    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch required to load checkpoint.pt") from exc

    try:
        ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(pt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "embeddings" in ckpt and "labels" in ckpt:
            emb = ckpt["embeddings"]
            labels = ckpt["labels"]
        elif "student" in ckpt and hasattr(ckpt["student"], "last_embeddings"):
            emb = ckpt["student"].last_embeddings
            labels = ckpt["student"].last_labels
        else:
            raise KeyError(f"Unrecognized checkpoint layout in {pt_path}")
    else:
        raise TypeError(f"Expected dict checkpoint at {pt_path}")

    if hasattr(emb, "detach"):
        emb = emb.detach().cpu().numpy()
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()
    return np.asarray(emb), np.asarray(labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--saot-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "embeddings" / "comparison.npz")
    args = parser.parse_args()

    b_emb, b_lab = load_embeddings(args.baseline_run)
    s_emb, s_lab = load_embeddings(args.saot_run)
    if b_lab.shape != s_lab.shape or not np.array_equal(b_lab, s_lab):
        raise ValueError("Baseline and SAOT label arrays must match (same nodes / ordering)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        baseline_emb=b_emb.astype(np.float32),
        saot_emb=s_emb.astype(np.float32),
        labels=b_lab,
    )
    print(f"Wrote {args.out} with {b_emb.shape[0]} nodes")


if __name__ == "__main__":
    main()
