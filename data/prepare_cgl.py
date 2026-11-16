#!/usr/bin/env python3
"""Prepare deterministic continual-learning task splits and validate them with a GCN.

Inputs
------
* CoraFull-CL: ``data/raw/CoraFull/cora.npz``. If absent, the script downloads
  the public CoraFull archive used by PyTorch Geometric.
* Arxiv-CL: ``data/raw/Arxiv/arxiv/raw`` from the official OGB ``arxiv.zip``.
  If absent, the script downloads and extracts the archive.

Outputs
-------
* ``data/task_splits/<dataset>_seed<seed>.json``
* ``data/data_statistics.csv``
* ``results/figures/member_1/task_split_overview.png``
* Optional GCN validation results under
  ``results/gcn_sequential/CoraFull-CL/<scenario>/seed_<seed>/``.

Shared interface
----------------
``load_graph(dataset, seed)`` returns a ``torch_geometric.data.Data`` object
with global zero-based node IDs.  Task JSON files expose both the detailed
``tasks`` records and the top-level ``task_to_*`` mappings consumed by members
3, 4, and 5.

Commands
--------
Prepare one dataset (acceptance command):
``python data/prepare_cgl.py --dataset CoraFull-CL --seed 42``

Prepare both datasets and run the member-1 independent experiment:
``python data/prepare_cgl.py --dataset all --seed 42 --run-gcn-validation``

The NumPy/SciPy implementation runs on CPU and requires no GPU. A CUDA GCN
implementation used by the integrated project is expected to need less than
2 GB on CoraFull-CL and about 4-8 GB on Arxiv-CL, depending on sampling.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
import math
import pickle
import shutil
import sys
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "example.yaml"
CORA_URL = "https://github.com/abojchevski/graph2gauss/raw/master/data/cora.npz"
ARXIV_URL = "https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip"
DATASET_ALIASES = {
    "corafull-cl": "CoraFull-CL",
    "corafull": "CoraFull-CL",
    "cora": "CoraFull-CL",
    "arxiv-cl": "Arxiv-CL",
    "ogbn-arxiv": "Arxiv-CL",
    "arxiv": "Arxiv-CL",
}


@dataclass
class GraphData:
    name: str
    labels: np.ndarray
    edge_index: np.ndarray
    features: sp.csr_matrix | None
    feature_dim: int
    source: str

    @property
    def num_nodes(self) -> int:
        return int(self.labels.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def num_classes(self) -> int:
        return int(np.unique(self.labels).size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--dataset",
        default=None,
        help="CoraFull-CL, Arxiv-CL, or all. Defaults to the config value.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-gcn-validation", action="store_true")
    parser.add_argument("--validation-epochs", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Fail instead of downloading a missing public dataset.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def canonical_dataset(name: str) -> str:
    if name.lower() == "all":
        return "all"
    try:
        return DATASET_ALIASES[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset: {name}") from exc


def download(url: str, destination: Path, force: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        return
    logging.info("Downloading %s", url)
    with urllib.request.urlopen(url) as response, destination.open("wb") as stream:
        shutil.copyfileobj(response, stream)


def load_corafull(no_download: bool, force_download: bool) -> GraphData:
    raw_path = REPO_ROOT / "data" / "raw" / "CoraFull" / "cora.npz"
    if not raw_path.exists() and no_download:
        raise FileNotFoundError(f"Missing {raw_path}")
    download(CORA_URL, raw_path, force=force_download)

    with np.load(raw_path, allow_pickle=True) as archive:
        adjacency = sp.csr_matrix(
            (
                archive["adj_data"],
                archive["adj_indices"],
                archive["adj_indptr"],
            ),
            shape=tuple(archive["adj_shape"]),
        )
        adjacency = adjacency.maximum(adjacency.T).tocsr()
        adjacency.setdiag(0)
        adjacency.eliminate_zeros()
        features = sp.csr_matrix(
            (
                archive["attr_data"],
                archive["attr_indices"],
                archive["attr_indptr"],
            ),
            shape=tuple(archive["attr_shape"]),
            dtype=np.float32,
        )
        labels = archive["labels"].astype(np.int64)

    coo = adjacency.tocoo()
    edge_index = np.vstack((coo.row, coo.col)).astype(np.int64)
    return GraphData(
        name="CoraFull-CL",
        labels=labels,
        edge_index=edge_index,
        features=features,
        feature_dim=int(features.shape[1]),
        source=CORA_URL,
    )


def _count_gzip_rows(path: Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def _load_gzip_ints(path: Path, expected_columns: int = 1) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",", dtype=np.int64)
    if expected_columns == 1:
        return values.reshape(-1)
    values = np.atleast_2d(values)
    if values.shape[1] != expected_columns:
        raise ValueError(f"Expected {expected_columns} columns in {path}")
    return values


def load_arxiv(no_download: bool, force_download: bool) -> GraphData:
    raw_root = REPO_ROOT / "data" / "raw" / "Arxiv"
    zip_path = raw_root / "arxiv.zip"
    extracted_root = raw_root / "arxiv" / "raw"
    label_path = extracted_root / "node-label.csv.gz"
    if (not label_path.exists() or force_download) and no_download:
        raise FileNotFoundError(f"Missing official OGB archive under {raw_root}")
    download(ARXIV_URL, zip_path, force=force_download)
    if not label_path.exists() or force_download:
        logging.info("Extracting %s", zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(raw_root)

    labels = _load_gzip_ints(label_path)
    edges = _load_gzip_ints(extracted_root / "edge.csv.gz", expected_columns=2).T
    return GraphData(
        name="Arxiv-CL",
        labels=labels,
        edge_index=edges,
        features=None,
        feature_dim=128,
        source=ARXIV_URL,
    )


def load_graph(
    dataset: str = "CoraFull-CL",
    seed: int = 42,
    *,
    no_download: bool = False,
    force_download: bool = False,
):
    """Return the global graph as ``torch_geometric.data.Data``.

    Node IDs are the original, global, zero-based IDs.  ``edge_index`` uses
    COO layout and CoraFull is explicitly stored in both directions.  The
    ``seed`` argument is accepted for the shared member-3/member-5 loader
    contract; graph loading itself is deterministic and does not consume it.

    The import is intentionally lazy so task split generation can run in a
    lightweight NumPy/SciPy environment.  Full graph loading requires PyTorch
    and PyTorch Geometric.
    """

    del seed
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as exc:
        raise ImportError(
            "load_graph requires torch and torch_geometric; install the shared "
            "project environment before integrating with members 3, 4, or 5."
        ) from exc

    name = canonical_dataset(dataset)
    if name == "all":
        raise ValueError("load_graph accepts one dataset, not 'all'")
    graph = (
        load_corafull(no_download, force_download)
        if name == "CoraFull-CL"
        else load_arxiv(no_download, force_download)
    )

    if graph.features is not None:
        features = graph.features.toarray().astype(np.float32, copy=False)
    else:
        feature_path = (
            REPO_ROOT
            / "data"
            / "raw"
            / "Arxiv"
            / "arxiv"
            / "raw"
            / "node-feat.csv.gz"
        )
        if not feature_path.exists():
            raise FileNotFoundError(f"Missing Arxiv node features: {feature_path}")
        features = np.loadtxt(feature_path, delimiter=",", dtype=np.float32)

    if features.shape != (graph.num_nodes, graph.feature_dim):
        raise ValueError(
            f"feature matrix has shape {features.shape}; expected "
            f"({graph.num_nodes}, {graph.feature_dim})"
        )

    return Data(
        x=torch.from_numpy(features).to(dtype=torch.float32),
        edge_index=torch.from_numpy(graph.edge_index).to(dtype=torch.long),
        y=torch.from_numpy(graph.labels).to(dtype=torch.long),
        num_nodes=graph.num_nodes,
    )


# Member 5 probes these conventional loader names in order.  Keeping aliases
# avoids adapter code while retaining ``load_graph`` as the documented API.
load_dataset = load_graph
build_graph = load_graph


def stratified_node_split(
    node_ids: np.ndarray, labels: np.ndarray, seed: int, task_id: int
) -> dict[str, list[int]]:
    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for label in sorted(np.unique(labels[node_ids]).tolist()):
        class_nodes = node_ids[labels[node_ids] == label].copy()
        class_rng = np.random.default_rng(seed + task_id * 1009 + int(label) * 9176)
        class_rng.shuffle(class_nodes)
        count = class_nodes.size
        train_end = max(1, int(math.floor(count * 0.60)))
        validation_end = max(train_end + 1, int(math.floor(count * 0.80)))
        validation_end = min(validation_end, max(1, count - 1))
        train_parts.append(class_nodes[:train_end])
        validation_parts.append(class_nodes[train_end:validation_end])
        test_parts.append(class_nodes[validation_end:])

    split_rng = np.random.default_rng(seed + task_id * 7919)
    result: dict[str, list[int]] = {}
    for name, parts in (
        ("train", train_parts),
        ("validation", validation_parts),
        ("test", test_parts),
    ):
        combined = np.concatenate(parts).astype(np.int64)
        split_rng.shuffle(combined)
        result[name] = combined.tolist()
    return result


def build_tasks(graph: GraphData, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    classes = np.unique(graph.labels).astype(np.int64)
    if classes.size % 2:
        raise ValueError(f"{graph.name} has an odd number of classes")
    class_order = np.random.default_rng(seed).permutation(classes)
    source = graph.edge_index[0]
    target = graph.edge_index[1]
    node_task = np.full(graph.num_nodes, -1, dtype=np.int32)

    task_nodes: list[np.ndarray] = []
    for task_id, pair in enumerate(class_order.reshape(-1, 2)):
        ids = np.flatnonzero(np.isin(graph.labels, pair))
        node_task[ids] = task_id
        task_nodes.append(ids)

    same_task_edges = node_task[source] == node_task[target]
    statistics: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    seen_classes: list[int] = []
    for task_id, (pair, node_ids) in enumerate(
        zip(class_order.reshape(-1, 2), task_nodes)
    ):
        seen_classes.extend(int(value) for value in pair)
        splits = stratified_node_split(node_ids, graph.labels, seed, task_id)
        edge_mask = same_task_edges & (node_task[source] == task_id)
        incident = (node_task[source] == task_id) | (node_task[target] == task_id)
        removed_cross_edges = int(np.count_nonzero(incident & ~same_task_edges))
        counts = {str(int(label)): int(np.count_nonzero(graph.labels[node_ids] == label)) for label in pair}
        task = {
            "task_id": task_id,
            "classes": [int(value) for value in pair],
            "task_il_label_map": {str(int(pair[0])): 0, str(int(pair[1])): 1},
            "class_il_seen_classes": sorted(seen_classes),
            "node_ids": node_ids.astype(np.int64).tolist(),
            "split_node_ids": splits,
            "statistics": {
                "num_nodes": int(node_ids.size),
                "num_train": len(splits["train"]),
                "num_validation": len(splits["validation"]),
                "num_test": len(splits["test"]),
                "num_intra_task_directed_edges": int(np.count_nonzero(edge_mask)),
                "num_cross_task_directed_edges_removed": removed_cross_edges,
                "class_counts": counts,
            },
        }
        tasks.append(task)
        statistics.append(
            {
                "dataset": graph.name,
                "seed": seed,
                "task_id": task_id,
                "class_1": int(pair[0]),
                "class_2": int(pair[1]),
                "num_nodes": int(node_ids.size),
                "num_train": len(splits["train"]),
                "num_validation": len(splits["validation"]),
                "num_test": len(splits["test"]),
                "num_intra_task_directed_edges": int(np.count_nonzero(edge_mask)),
                "num_cross_task_directed_edges_removed": removed_cross_edges,
            }
        )

    task_to_classes = {
        str(task["task_id"]): task["classes"] for task in tasks
    }
    task_to_node_ids = {
        str(task["task_id"]): task["node_ids"] for task in tasks
    }
    task_to_train_ids = {
        str(task["task_id"]): task["split_node_ids"]["train"] for task in tasks
    }
    task_to_val_ids = {
        str(task["task_id"]): task["split_node_ids"]["validation"]
        for task in tasks
    }
    task_to_test_ids = {
        str(task["task_id"]): task["split_node_ids"]["test"] for task in tasks
    }
    num_edges_removed = int(np.count_nonzero(~same_task_edges))
    payload = {
        "schema_version": 1,
        "dataset": graph.name,
        "seed": seed,
        "task_definition": "two disjoint classes per task; inter-task edges excluded",
        "class_order": [int(value) for value in class_order],
        "num_nodes": graph.num_nodes,
        "num_directed_edges": graph.num_edges,
        "num_classes": graph.num_classes,
        "num_tasks": len(tasks),
        "feature_dim": graph.feature_dim,
        "source": graph.source,
        "num_edges_removed": num_edges_removed,
        "task_to_classes": task_to_classes,
        "task_to_node_ids": task_to_node_ids,
        "task_to_train_ids": task_to_train_ids,
        "task_to_val_ids": task_to_val_ids,
        "task_to_test_ids": task_to_test_ids,
        "tasks": tasks,
    }
    return payload, statistics


def write_json_deterministic(payload: dict[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(serialized, encoding="utf-8", newline="\n")
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def write_statistics(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "seed",
        "task_id",
        "class_1",
        "class_2",
        "num_nodes",
        "num_train",
        "num_validation",
        "num_test",
        "num_intra_task_directed_edges",
        "num_cross_task_directed_edges_removed",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_task_overview(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    datasets = list(dict.fromkeys(row["dataset"] for row in rows))
    figure, axes = plt.subplots(len(datasets), 1, figsize=(12, 4.2 * len(datasets)), squeeze=False)
    palette = {"CoraFull-CL": "#315A7D", "Arxiv-CL": "#B55D3D"}
    for axis, dataset in zip(axes[:, 0], datasets):
        subset = [row for row in rows if row["dataset"] == dataset]
        x = np.arange(1, len(subset) + 1)
        totals = np.asarray([int(row["num_nodes"]) for row in subset], dtype=np.int64)
        axis.bar(x, totals, color=palette.get(dataset, "#4F6D7A"), width=0.82)
        axis.axhline(totals.mean(), color="#222222", linestyle="--", linewidth=1, label=f"mean = {totals.mean():.1f}")
        axis.set_title(f"{dataset} task split, seed 42")
        axis.set_ylabel("Nodes")
        axis.set_xlim(0.2, len(subset) + 0.8)
        axis.grid(axis="y", alpha=0.22)
        axis.legend(frameon=False, loc="upper right")
        step = 5 if len(subset) > 25 else 2
        axis.set_xticks(x[::step])
        axis.set_xticklabels([str(int(value)) for value in x[::step]])
    axes[-1, 0].set_xlabel("Task ID")
    figure.suptitle("Two-class continual graph learning task sizes", fontsize=15, y=0.995)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def normalize_features(features: sp.csr_matrix) -> sp.csr_matrix:
    sums = np.asarray(features.sum(axis=1)).reshape(-1)
    inverse = np.divide(1.0, sums, out=np.zeros_like(sums, dtype=np.float32), where=sums != 0)
    return sp.diags(inverse) @ features


def normalized_adjacency(edge_index: np.ndarray, node_ids: np.ndarray, num_nodes: int) -> sp.csr_matrix:
    local = np.full(num_nodes, -1, dtype=np.int64)
    local[node_ids] = np.arange(node_ids.size)
    source = local[edge_index[0]]
    target = local[edge_index[1]]
    mask = (source >= 0) & (target >= 0)
    adjacency = sp.csr_matrix(
        (np.ones(np.count_nonzero(mask), dtype=np.float32), (source[mask], target[mask])),
        shape=(node_ids.size, node_ids.size),
    )
    adjacency = adjacency.maximum(adjacency.T)
    adjacency = adjacency + sp.eye(node_ids.size, dtype=np.float32, format="csr")
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    inv_sqrt = np.power(degree, -0.5, where=degree > 0)
    return (sp.diags(inv_sqrt) @ adjacency @ sp.diags(inv_sqrt)).tocsr()


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def local_positions(node_ids: np.ndarray, selected_global_ids: Iterable[int], num_nodes: int) -> np.ndarray:
    mapping = np.full(num_nodes, -1, dtype=np.int64)
    mapping[node_ids] = np.arange(node_ids.size)
    return mapping[np.asarray(list(selected_global_ids), dtype=np.int64)]


def forward(
    features: sp.csr_matrix,
    adjacency: sp.csr_matrix,
    weight_0: np.ndarray,
    weight_1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    support = features @ weight_0
    pre_activation = adjacency @ support
    hidden = np.maximum(pre_activation, 0.0)
    logits = adjacency @ hidden @ weight_1
    return logits, hidden, pre_activation


def evaluate_task(
    graph: GraphData,
    task: dict[str, Any],
    weight_0: np.ndarray,
    weight_1: np.ndarray,
    seen_classes: list[int],
) -> tuple[float, float]:
    node_ids = np.asarray(task["node_ids"], dtype=np.int64)
    features = normalize_features(graph.features[node_ids].tocsr())
    adjacency = normalized_adjacency(graph.edge_index, node_ids, graph.num_nodes)
    logits, _, _ = forward(features, adjacency, weight_0, weight_1)
    test_positions = local_positions(
        node_ids, task["split_node_ids"]["test"], graph.num_nodes
    )
    labels = graph.labels[node_ids][test_positions]
    class_indices = np.asarray(seen_classes, dtype=np.int64)
    class_predictions = class_indices[np.argmax(logits[test_positions][:, class_indices], axis=1)]
    task_classes = np.asarray(task["classes"], dtype=np.int64)
    task_predictions = task_classes[np.argmax(logits[test_positions][:, task_classes], axis=1)]
    return float(np.mean(class_predictions == labels)), float(np.mean(task_predictions == labels))


def compute_ap_af(matrix: np.ndarray) -> tuple[float, float]:
    tasks = matrix.shape[0]
    final_values = matrix[tasks - 1, :]
    ap = float(np.nanmean(final_values))
    if tasks == 1:
        return ap, 0.0
    diagonal = np.diag(matrix)[:-1]
    af = float(np.nanmean(final_values[:-1] - diagonal))
    return ap, af


def write_matrix(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["after_task"] + [f"eval_task_{index}" for index in range(1, matrix.shape[1] + 1)])
        for index, row in enumerate(matrix, start=1):
            writer.writerow([index] + ["" if np.isnan(value) else f"{value:.6f}" for value in row])


def run_gcn_validation(
    graph: GraphData,
    split_payload: dict[str, Any],
    seed: int,
    epochs: int,
    hidden_dim: int,
    learning_rate: float,
) -> None:
    if graph.features is None:
        raise ValueError("GCN validation is implemented for CoraFull-CL")
    rng = np.random.default_rng(seed)
    weight_0 = rng.normal(0.0, 0.04, size=(graph.feature_dim, hidden_dim)).astype(np.float32)
    weight_1 = rng.normal(0.0, 0.04, size=(hidden_dim, graph.num_classes)).astype(np.float32)
    tasks = split_payload["tasks"]
    class_matrix = np.full((len(tasks), len(tasks)), np.nan, dtype=np.float32)
    task_matrix = np.full_like(class_matrix, np.nan)
    log_lines = [
        f"dataset={graph.name} seed={seed} tasks={len(tasks)} classes={graph.num_classes}",
        f"nodes={graph.num_nodes} directed_edges={graph.num_edges} feature_dim={graph.feature_dim}",
        f"model=2-layer GCN backend=numpy-scipy hidden_dim={hidden_dim} epochs_per_task={epochs} lr={learning_rate}",
    ]
    start = time.perf_counter()
    seen_classes: list[int] = []
    for row, task in enumerate(tasks):
        node_ids = np.asarray(task["node_ids"], dtype=np.int64)
        features = normalize_features(graph.features[node_ids].tocsr())
        adjacency = normalized_adjacency(graph.edge_index, node_ids, graph.num_nodes)
        train_positions = local_positions(
            node_ids, task["split_node_ids"]["train"], graph.num_nodes
        )
        labels = graph.labels[node_ids]
        seen_classes.extend(task["classes"])
        active_classes = np.asarray(seen_classes, dtype=np.int64)
        active_lookup = {label: index for index, label in enumerate(seen_classes)}
        final_loss = float("nan")
        for _ in range(epochs):
            logits, hidden, pre_activation = forward(features, adjacency, weight_0, weight_1)
            probabilities = softmax(logits[train_positions][:, active_classes])
            target_labels = labels[train_positions]
            target_positions = np.asarray(
                [active_lookup[int(label)] for label in target_labels], dtype=np.int64
            )
            final_loss = float(
                -np.log(
                    probabilities[np.arange(target_labels.size), target_positions] + 1e-12
                ).mean()
            )
            gradient_logits = np.zeros_like(logits, dtype=np.float32)
            gradient_train = probabilities
            gradient_train[np.arange(target_labels.size), target_positions] -= 1.0
            gradient_train /= target_labels.size
            gradient_logits[
                train_positions[:, None], active_classes[None, :]
            ] = gradient_train
            gradient_weight_1 = hidden.T @ (adjacency.T @ gradient_logits)
            gradient_hidden = adjacency.T @ (gradient_logits @ weight_1.T)
            gradient_pre = gradient_hidden * (pre_activation > 0)
            gradient_weight_0 = features.T @ (adjacency.T @ gradient_pre)
            gradient_weight_0 = np.asarray(gradient_weight_0, dtype=np.float32)
            weight_0 -= learning_rate * np.clip(gradient_weight_0, -1.0, 1.0)
            weight_1 -= learning_rate * np.clip(gradient_weight_1, -1.0, 1.0)

        for column in range(row + 1):
            class_acc, task_acc = evaluate_task(
                graph, tasks[column], weight_0, weight_1, seen_classes
            )
            class_matrix[row, column] = class_acc
            task_matrix[row, column] = task_acc
        class_ap = float(np.nanmean(class_matrix[row, : row + 1]))
        task_ap = float(np.nanmean(task_matrix[row, : row + 1]))
        message = (
            f"task={row + 1:02d}/{len(tasks)} classes={task['classes']} "
            f"loss={final_loss:.4f} class_il_ap={class_ap:.4f} task_il_ap={task_ap:.4f}"
        )
        print(message)
        log_lines.append(message)

    elapsed = time.perf_counter() - start
    class_ap, class_af = compute_ap_af(class_matrix)
    task_ap, task_af = compute_ap_af(task_matrix)
    summary = (
        f"completed elapsed_seconds={elapsed:.2f} class_il_ap={class_ap:.6f} "
        f"class_il_af={class_af:.6f} task_il_ap={task_ap:.6f} task_il_af={task_af:.6f}"
    )
    print(summary)
    log_lines.append(summary)

    common_config = {
        "dataset": graph.name,
        "seed": seed,
        "model": "gcn_sequential_numpy_validation",
        "hidden_dim": hidden_dim,
        "epochs_per_task": epochs,
        "learning_rate": learning_rate,
        "task_split_sha256": hashlib.sha256(
            json.dumps(split_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    metrics = {
        "class_il": {"AP": class_ap, "AF": class_af},
        "task_il": {"AP": task_ap, "AF": task_af},
        "elapsed_seconds": elapsed,
        "note": "Member-1 supervised GCN smoke test for validating the data pipeline; not a SAOT paper reproduction result.",
    }
    checkpoint = {
        "format": "pickle with NumPy arrays",
        "weight_0": weight_0,
        "weight_1": weight_1,
        "config": common_config,
    }
    for scenario, matrix in (("class-il", class_matrix), ("task-il", task_matrix)):
        output_dir = REPO_ROOT / "results" / "gcn_sequential" / graph.name / scenario / f"seed_{seed}"
        output_dir.mkdir(parents=True, exist_ok=True)
        config = dict(common_config, scenario=scenario)
        (output_dir / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_matrix(output_dir / "performance_matrix.csv", matrix)
        (output_dir / "train.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        with (output_dir / "checkpoint.pt").open("wb") as stream:
            pickle.dump(checkpoint, stream, protocol=pickle.HIGHEST_PROTOCOL)

    log_dir = REPO_ROOT / "logs" / "member_1" / f"gcn_sequential_seed{seed}"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "train.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    config = load_config(args.config)
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    requested = canonical_dataset(args.dataset or str(config.get("dataset", "CoraFull-CL")))
    names = ["CoraFull-CL", "Arxiv-CL"] if requested == "all" else [requested]
    all_statistics: list[dict[str, Any]] = []
    prepared: dict[str, tuple[GraphData, dict[str, Any]]] = {}
    for name in names:
        graph = (
            load_corafull(args.no_download, args.force_download)
            if name == "CoraFull-CL"
            else load_arxiv(args.no_download, args.force_download)
        )
        payload, statistics = build_tasks(graph, seed)
        filename = "corafull" if name == "CoraFull-CL" else "arxiv"
        json_path = REPO_ROOT / "data" / "task_splits" / f"{filename}_seed{seed}.json"
        digest = write_json_deterministic(payload, json_path)
        print(
            f"{name}: nodes={graph.num_nodes} directed_edges={graph.num_edges} "
            f"classes={graph.num_classes} tasks={payload['num_tasks']} sha256={digest}"
        )
        all_statistics.extend(statistics)
        prepared[name] = (graph, payload)

    statistics_path = REPO_ROOT / "data" / "data_statistics.csv"
    existing_rows: list[dict[str, Any]] = []
    if statistics_path.exists():
        with statistics_path.open("r", encoding="utf-8-sig", newline="") as stream:
            existing_rows = list(csv.DictReader(stream))
    retained = [row for row in existing_rows if row.get("dataset") not in names or int(row.get("seed", -1)) != seed]
    write_statistics(retained + all_statistics, statistics_path)
    plot_rows = retained + all_statistics
    plot_task_overview(plot_rows, REPO_ROOT / "results" / "figures" / "member_1" / "task_split_overview.png")

    if args.run_gcn_validation:
        if "CoraFull-CL" not in prepared:
            graph = load_corafull(args.no_download, args.force_download)
            payload, _ = build_tasks(graph, seed)
        else:
            graph, payload = prepared["CoraFull-CL"]
        run_gcn_validation(
            graph,
            payload,
            seed=seed,
            epochs=args.validation_epochs,
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
