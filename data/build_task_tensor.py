"""Convert prepared JSON task metadata into tensors consumed by the SAOT trainer.

The preparation script intentionally stores deterministic, lightweight metadata.
This module materializes one local graph per task and keeps the original global
labels so Class-IL and Task-IL probes can be evaluated without reconstructing
the raw graph during training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import gzip
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.prepare_cgl import canonical_dataset, load_arxiv, load_corafull


def _local_edges(global_edges: np.ndarray, node_ids: np.ndarray, num_nodes: int) -> torch.Tensor:
    mapping = np.full(num_nodes, -1, dtype=np.int64)
    mapping[node_ids] = np.arange(node_ids.size, dtype=np.int64)
    source = mapping[global_edges[0]]
    target = mapping[global_edges[1]]
    keep = (source >= 0) & (target >= 0)
    return torch.from_numpy(np.vstack((source[keep], target[keep])).astype(np.int64, copy=False))


def _mask(node_ids: np.ndarray, selected: list[int]) -> torch.Tensor:
    selected_set = set(int(value) for value in selected)
    return torch.tensor([int(value) in selected_set for value in node_ids], dtype=torch.bool)


def build_task_payload(split_payload: dict[str, Any], graph: Any) -> dict[str, Any]:
    if graph.features is not None:
        features = graph.features.toarray().astype(np.float32, copy=False)
    else:
        feature_path = ROOT / "data" / "raw" / "Arxiv" / "arxiv" / "raw" / "node-feat.csv.gz"
        if not feature_path.exists():
            raise FileNotFoundError(f"missing Arxiv node features: {feature_path}")
        with gzip.open(feature_path, "rt", encoding="utf-8") as stream:
            features = np.loadtxt(stream, delimiter=",", dtype=np.float32)
        if features.shape != (graph.num_nodes, graph.feature_dim):
            raise ValueError(f"Arxiv features have shape {features.shape}; expected {(graph.num_nodes, graph.feature_dim)}")
    tasks = []
    for record in split_payload["tasks"]:
        node_ids = np.asarray(record["node_ids"], dtype=np.int64)
        x = torch.from_numpy(np.ascontiguousarray(features[node_ids]))
        y = torch.from_numpy(np.asarray(graph.labels[node_ids], dtype=np.int64))
        edge_index = _local_edges(graph.edge_index, node_ids, graph.num_nodes)
        splits = record["split_node_ids"]
        tasks.append(
            {
                "task_id": int(record["task_id"]),
                "classes": torch.tensor(record["classes"], dtype=torch.long),
                "node_ids": torch.from_numpy(node_ids),
                "x": x,
                "y": y,
                "edge_index": edge_index,
                "train_mask": _mask(node_ids, splits["train"]),
                "validation_mask": _mask(node_ids, splits["validation"]),
                "test_mask": _mask(node_ids, splits["test"]),
            }
        )
    return {
        "format": "saot_task_tensors_v1",
        "dataset": split_payload["dataset"],
        "seed": int(split_payload["seed"]),
        "split_schema_version": int(split_payload.get("schema_version", 1)),
        "split_sha256": hashlib.sha256(
            json.dumps(split_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "feature_dim": int(graph.feature_dim),
        "num_tasks": len(tasks),
        "tasks": tasks,
    }


def build_task_tensor(dataset: str, split_path: Path, output_path: Path, *, no_download: bool = False) -> Path:
    name = canonical_dataset(dataset)
    if name == "all":
        raise ValueError("build_task_tensor accepts one dataset at a time")
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    if split_payload.get("dataset") != name:
        raise ValueError(f"split file dataset={split_payload.get('dataset')} does not match {name}")
    graph = load_corafull(no_download, False) if name == "CoraFull-CL" else load_arxiv(no_download, False)
    payload = build_task_payload(split_payload, graph)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(digest + "\n", encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    output = build_task_tensor(args.dataset, args.split, args.output, no_download=args.no_download)
    print(f"wrote {output}")
    print(f"sha256 {output.with_suffix(output.suffix + '.sha256').read_text().strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
